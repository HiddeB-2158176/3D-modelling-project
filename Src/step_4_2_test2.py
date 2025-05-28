import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import GrayCodeEncoder
import random
import open3d as o3d
from scipy.spatial.transform import Rotation as ScipyRotation 
from scipy.spatial.transform import Slerp

from step_2 import decode_gray_pattern, find_correspondences
from step_3 import compute_essential_matrix, triangulate_opencv, triangulate_manual, get_colors

# This file was our first decent try of the plane sweep algorithm.
# We did try to follow the steps in the assigenment here,
# but as you can see in the Result folder, the output isn't what we are looking for.
# In the other files we tried different approaches to the plane sweep algorithm,
# some of them following the steps in the assignment more closely than others.

# Create a depth map from the 3D points
def create_depth_map(points_3D, colors, img_shape, pts1):
    """
    Create a depth map from the 3D points.

    :param points_3D: Nx3 array of 3D points.
    :param colors: Nx3 array of RGB colors corresponding to the 3D points.
    :param img_shape: Shape of the image (height, width).
    :param pts1: Nx2 array of 2D points in the first image.
    :return: Depth map as a 2D array.
    """
    print("Image shape:", img_shape)
    depth_map = np.zeros(img_shape[:2], dtype=np.float32)

    for (x, y), color, point in zip(pts1.astype(int), colors, points_3D):
        if 0 <= y < img_shape[0] and 0 <= x < img_shape[1]:
            depth_map[y, x] = point[2]
    
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return depth_map

def visualize_depth_map(depthMap):
    """
    Visualize the depth map.

    :param depthMap: 2D array representing the depth map.
    """
    plt.imshow(depthMap, cmap='gray')
    plt.colorbar()
    plt.show()

def compute_scene_center_depth(points_3d, projection_matrix):
    """
    Computes the center of the point cloud and its depth in the camera view.

    :param points_3d: Nx3 array of 3D points in world coordinates.
    :param projection_matrix: 3x4 camera projection matrix.
    :return: Depth of the center point in the camera view.
    """
    # Compute the centroid of the point cloud
    centroid = np.mean(points_3d, axis=0)  

    # Convert to homogeneous coordinates
    centroid_h = np.append(centroid, 1) 

    # Project the centroid using the projection matrix
    projected = projection_matrix @ centroid_h  

    # Convert from homogeneous if necessary
    if projected.shape[0] == 4:
        depth = projected[2] / projected[3]
    else:
        depth = projected[2] 

    return depth

def deproject_image_corners(intrinsics, extrinsics, width, height, depth):
    """
    Deprojects the image plane corners into 3D space at a given depth.

    :param intrinsics: 3x3 camera intrinsics matrix.
    :param extrinsics: 4x4 world-to-camera transformation matrix.
    :param width: Width of the image in pixels.
    :param height: Height of the image in pixels.
    :param depth: Depth at which to deproject the corners.
    :return: 4x3 array of 3D points in world coordinates.
    """
    # Define 2D image corners in pixel coordinates
    image_corners = np.array([
        [0, 0],             
        [width - 1, 0],      
        [0, height - 1],     
        [width - 1, height - 1] 
    ])  # shape: (4, 2)

    # Convert corners to normalized camera coordinates
    inv_K = np.linalg.inv(intrinsics)
    corners_homog = np.concatenate([image_corners, np.ones((4, 1))], axis=1).T  # shape: (3, 4)
    rays = inv_K @ corners_homog  # shape: (3, 4)

    # Scale rays to the given depth
    rays = rays * depth / rays[2, :]  # normalize z to depth

    # Convert to world coordinates using extrinsics
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_3d = (R @ rays + t[:, np.newaxis]).T  # shape: (4, 3)

    return points_3d

def project_points_to_image(points_3d, intrinsics, extrinsics):
    """
    Projects 3D points onto the image plane using intrinsics and extrinsics.

    :param points_3d: Nx3 array of 3D points in world coordinates.
    :param intrinsics: 3x3 camera intrinsics matrix.
    :param extrinsics: 4x4 world-to-camera transformation matrix.
    :return: Nx2 array of 2D points in image coordinates.
    """
    # Convert to homogeneous coordinates
    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # (4, 4)
    proj_matrix = intrinsics @ extrinsics[:3, :]  # 3x4

    projected = proj_matrix @ points_3d_h.T  # (3, 4)
    projected /= projected[2, :]  # Normalize
    return projected[:2, :].T  # (4, 2)


def warp_image_to_depth_plane(image, depth_plane_3d, camera_intrinsics, camera_extrinsics):
    """
    Warps the input image to align with the 3D depth plane using homography.

   :param image: Input image to be warped.
    :param depth_plane_3d: 4x3 array of 3D points representing the depth plane corners.
    :param camera_intrinsics: 3x3 camera intrinsics matrix.
    :param camera_extrinsics: 4x4 world-to-camera transformation matrix.
    :return: Warped image aligned with the depth plane.
    """
    h, w = image.shape[:2]

    # Image corners (2D)
    image_corners = np.array([
        [0, 0],         # top-left
        [w - 1, 0],     # top-right
        [0, h - 1],     # bottom-left
        [w - 1, h - 1]  # bottom-right
    ], dtype=np.float32)

    # Project 3D plane corners to input camera image
    projected_2d = project_points_to_image(depth_plane_3d, camera_intrinsics, camera_extrinsics).astype(np.float32)

    # Compute homography from input image to projected plane
    H, _ = cv2.findHomography(image_corners, projected_2d)

    # Warp image using the homography
    warped_image = cv2.warpPerspective(image, H, (w, h))

    return warped_image


def compute_best_depth_image(warped_left_list, warped_right_list):
    """
    Computes the final interpolated image by minimizing pixel-wise error across depth layers.

    :param warped_left_list: List of left images warped to virtual camera.
    :param warped_right_list: List of right images warped to virtual camera.
    :return: Final rendered image from virtual viewpoint.
    """
    num_layers = len(warped_left_list)
    h, w, c = warped_left_list[0].shape

    min_error = np.full((h, w), np.inf)
    best_image = np.zeros((h, w, c), dtype=np.uint8)

    for i in range(num_layers):
        left = warped_left_list[i]
        right = warped_right_list[i]

        # Convert to grayscale
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Apply slight blur to reduce noise
        gray_left = cv2.GaussianBlur(gray_left, (3, 3), 0)
        gray_right = cv2.GaussianBlur(gray_right, (3, 3), 0)

        # Absolute pixel-wise difference
        error = cv2.absdiff(gray_left, gray_right).astype(np.float32)

        # Update mask where error improves
        mask = error < min_error
        min_error[mask] = error[mask]

        # Average left and right color images where mask is valid
        blended = ((left.astype(np.uint16) + right.astype(np.uint16)) // 2).astype(np.uint8)
        best_image[mask] = blended[mask]

    return best_image

def apply_transform(points_3d, transform):
    """
    Applies a transformation matrix to a set of 3D points.

    :param points_3d: Nx3 array of 3D points in world coordinates.
    :param transform: 4x4 transformation matrix (rotation + translation).
    :return: Nx3 array of transformed 3D points.
    """

    num_pts = points_3d.shape[0]
    homog = np.hstack([points_3d, np.ones((num_pts, 1))])
    transformed = (transform @ homog.T).T
    return transformed[:, :3]


def make_warped_lists(K, R, T, img_left, img_right, width, height, depth_center):
    """
    Creates lists of warped images for a virtual camera at various depths.

    :param K: Camera intrinsics matrix.
    :param R: Rotation matrix from right camera to virtual camera.
    :param T: Translation vector from left camera to right camera.
    :param img_left: Left camera image.
    :param img_right: Right camera image.
    :param width: Width of the images.
    :param height: Height of the images.
    :param depth_center: Center depth for the virtual camera.
    :return: Lists of warped left and right images.
    """

    warped_left_list = []
    warped_right_list = []

    extr_left = np.eye(4)
    extr_left[:3, :3] = np.eye(3)  
    extr_left[:3, 3] = np.zeros(3) 

    extr_right = np.eye(4)
    extr_right[:3, :3] = R
    extr_right[:3, 3] = T.squeeze()

    midpoint_translation = (extr_left[:3, 3] + extr_right[:3, 3]) / 2

    extr_virtual = np.eye(4)
    extr_virtual[:3, :3] = np.eye(3) 
    extr_virtual[:3, 3] = midpoint_translation

    num_layers = 50
    depth_range = 0.1  
    depths = np.linspace(depth_center * (1 - depth_range),
                         depth_center * (1 + depth_range),
                         num_layers)
    
    black_count_left = 0
    black_count_right = 0
    for d in depths:
        # Deproject corners in virtual camera frame
        depth_plane_3d_virtual = deproject_image_corners(K, extr_virtual, width, height, d)

        # Compute relative transforms
        T_virtual_to_left = extr_left @ np.linalg.inv(extr_virtual)
        T_virtual_to_right = extr_right @ np.linalg.inv(extr_virtual)

        # Transform depth plane into left and right camera coordinates
        depth_plane_3d_left = apply_transform(depth_plane_3d_virtual, T_virtual_to_left)
        depth_plane_3d_right = apply_transform(depth_plane_3d_virtual, T_virtual_to_right)

        # Warp images to the virtual depth plane
        warped_left = warp_image_to_depth_plane(img_left, depth_plane_3d_left, K, extr_left)
        warped_right = warp_image_to_depth_plane(img_right, depth_plane_3d_right, K, extr_right)


        # Check if the warped images are empty or have unusual pixel values
        if np.all(warped_left == 0):
            black_count_left += 1 
            print(f"Warning: warped_left is all black at depth {d}")
        if np.all(warped_right == 0):
            black_count_right += 1
            print(f"Warning: warped_right is all black at depth {d}")

        cv2.imshow("Warped Left", warped_left)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Warped Right", warped_right)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        warped_left_list.append(warped_left)
        warped_right_list.append(warped_right)

    print(f"Number of black warped left images: {black_count_left}")
    print(f"Number of black warped right images: {black_count_right}")
    return warped_left_list, warped_right_list

def render_virtual_view(warped_left_list, warped_right_list):
    """
    Renders the final virtual camera image from warped left and right image stacks.

    :param warped_left_list: List of left images warped to virtual camera.
    :param warped_right_list: List of right images warped to virtual camera.
    :return: Final rendered image from virtual viewpoint.
    """
    num_layers = len(warped_left_list)
    h, w, c = warped_left_list[0].shape

    min_error = np.full((h, w), np.inf)
    best_image = np.zeros((h, w, c), dtype=np.uint8)

    for i in range(num_layers):
        left = warped_left_list[i]
        right = warped_right_list[i]

        # Compute pixel-wise error
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        error = cv2.absdiff(gray_left, gray_right).astype(np.float32)

        # Mask where this layer is better
        better_mask = error < min_error
        min_error[better_mask] = error[better_mask]

        # Average color
        blended = ((left.astype(np.uint16) + right.astype(np.uint16)) // 2).astype(np.uint8)

        # Update best image
        best_image[better_mask] = blended[better_mask]

    return best_image


if __name__ == "__main__":
    images_view0 = glob.glob('../Data/GrayCodes/view0/*.jpg')
    images_view1 = glob.glob('../Data/GrayCodes/view1/*.jpg')

    images0 = []
    images1 = []
    for i in range(len(images_view0)):
        resized0 = cv2.resize(cv2.imread(images_view0[i], cv2.IMREAD_GRAYSCALE), (1920, 1080))
        resized1 = cv2.resize(cv2.imread(images_view1[i], cv2.IMREAD_GRAYSCALE), (1920, 1080))
        images0.append(resized0)
        images1.append(resized1)

    result0 = decode_gray_pattern(images0)
    result1 = decode_gray_pattern(images1)

    img1 = images0[0]
    img2 = images1[0]

    img1Color = cv2.resize(cv2.imread(images_view0[0], cv2.IMREAD_COLOR), (1920, 1080))
    img2Color = cv2.resize(cv2.imread(images_view1[0], cv2.IMREAD_COLOR), (1920, 1080))

    img1ColorRGB = cv2.cvtColor(img1Color, cv2.COLOR_BGR2RGB)
    img2ColorRGB = cv2.cvtColor(img2Color, cv2.COLOR_BGR2RGB)

    keypoints1, keypoints2, matches = find_correspondences(result0, result1)

    K = np.array([[9.51663140e+03, 0.00000000e+00, 2.81762458e+03],[0.00000000e+00, 8.86527952e+03, 1.14812762e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    E, mask, pts1, pts2 = compute_essential_matrix(keypoints1, keypoints2, matches, K)

    # R = rotatie vector
    # T = translatie vector
    _, R, T, mask = cv2.recoverPose(E, pts1, pts2, K)

    points_3D_opencv = triangulate_opencv(pts1, pts2, K, R, T)
    #points_3D_manual = triangulate_manual(pts1, pts2, K, R, T)

    colors = get_colors(img1Color, img2Color, pts1, pts2)

    # Compute the projection matrix
    projection_matrix = np.hstack((R, T))
    projection_matrix = K @ projection_matrix  # Shape: (3, 4)

    # Compute the depth of the center point
    center_depth = compute_scene_center_depth(points_3D_opencv, projection_matrix)
    print(f"Depth of the center point: {center_depth:.2f}")

    # Deproject the image corners
    deprojected_corners = deproject_image_corners(K, np.hstack((R, T)), img1Color.shape[1], img1Color.shape[0], center_depth)
    print("Deprojected corners (in world coordinates):")
    print(deprojected_corners)

    # warp the image to the depth plane
    extr_left = np.eye(4)  # left camera at origin
    extr_right = np.eye(4)
    extr_right[:3, :3] = R
    extr_right[:3, 3] = T.squeeze()
    
    warped_left_list, warped_right_list = make_warped_lists(K, R, T, img1Color, img2Color, img1Color.shape[1], img1Color.shape[0], center_depth)

    final_output = compute_best_depth_image(warped_left_list, warped_right_list)
    cv2.imshow("Final Interpolated Image", final_output)
    cv2.waitKey(0)

    final_output = render_virtual_view(warped_left_list, warped_right_list)
    cv2.imshow("Final Rendered Image", final_output)
    cv2.waitKey(0)

    # save the image
    #cv2.imwrite("../Result/final_output_plane_sweep1.png", final_output)
    cv2.destroyAllWindows()

                                                                           

