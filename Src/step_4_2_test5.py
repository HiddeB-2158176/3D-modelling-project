import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp
import glob

from step_2 import decode_gray_pattern, find_correspondences
from step_3 import compute_essential_matrix


# This file contains functions for interpolating camera poses, projecting 3D points into 2D,
# computing depth maps using plane sweep stereo, and visualizing results with Open3D. 
# This file contains results that are similr to the results of step_4_2_test6.py, 
# but these are more unclear and not as good as the results of step_4_2_test6.py.
# We took a different approach to the plane sweep stereo algorithm,
# where we compute the cost volume using normalized cross-correlation (NCC) between patches

def interpolate_pose(R0, T0, R1, T1, alpha):
    """
    Interpolate between two camera poses using spherical linear interpolation (SLERP).

    :param R0: Rotation matrix of the first camera pose.
    :param T0: Translation vector of the first camera pose.
    :param R1: Rotation matrix of the second camera pose.
    :param T1: Translation vector of the second camera pose.
    :param alpha: Interpolation factor (0 <= alpha <= 1).
    :return: Interpolated rotation matrix and translation vector.
    """

    times = [0, 1]
    rotations = SciRotation.from_matrix([R0, R1])
    slerp = Slerp(times, rotations)
    r_interp = slerp([alpha])[0]
    T_interp = (1 - alpha) * T0 + alpha * T1
    return r_interp.as_matrix(), T_interp


def project_points(points_3d, K, R, T):
    """
    Project 3D points into 2D image plane using camera intrinsic matrix K and pose (R, T).

    :param points_3d: Nx3 array of 3D points in world coordinates.
    :param K: Camera intrinsic matrix (3x3).
    :param R: Rotation matrix (3x3) representing camera orientation.
    :param T: Translation vector (3,) representing camera position.
    :return: Nx2 array of projected 2D points in image coordinates.
    """

    P = K @ np.hstack((R, T.reshape(3, 1)))
    points_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    pts_2d_hom = P @ points_hom
    pts_2d = (pts_2d_hom[:2] / pts_2d_hom[2]).T
    return pts_2d


def normalized_cross_correlation(patch1, patch2, epsilon=1e-5):
    """
    Compute normalized cross-correlation (NCC) between two image patches.

    :param patch1: First image patch (H, W).
    :param patch2: Second image patch (H, W).
    :param epsilon: Small value to avoid division by zero.
    :return: NCC value (scalar).
    """

    mean1, mean2 = np.mean(patch1), np.mean(patch2)
    numerator = np.sum((patch1 - mean1) * (patch2 - mean2))
    denominator = np.sqrt(np.sum((patch1 - mean1)**2) * np.sum((patch2 - mean2)**2)) + epsilon
    return numerator / denominator


def plane_sweep_depth(img1, img2, K, R0, T0, R1, T1,
                      depth_min=0.5, depth_max=5.0, depth_steps=50, patch_size=7):
    """
    Perform plane sweep stereo to compute depth map between two images.

    :param img1: First grayscale image (H, W).
    :param img2: Second grayscale image (H, W).
    :param K: Camera intrinsic matrix (3x3).
    :param R0: Rotation matrix of the first camera pose (3x3).
    :param T0: Translation vector of the first camera pose (3,).
    :param R1: Rotation matrix of the second camera pose (3x3).
    :param T1: Translation vector of the second camera pose (3,).
    :param depth_min: Minimum depth value to consider.
    :param depth_max: Maximum depth value to consider.
    :param depth_steps: Number of depth steps to sample between min and max.
    :param patch_size: Size of the square patches to compare (must be odd).
    :return: Depth map (H, W) and cost volume (H, W).
    """
    
    h, w = img1.shape[:2]
    depth_map = np.zeros((h, w), dtype=np.float32)
    cost_volume = np.full((h, w), -np.inf, dtype=np.float32)

    half_patch = patch_size // 2
    alphas = np.linspace(0, 1, depth_steps)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    for i, alpha in enumerate(alphas):
        Rv, Tv = interpolate_pose(R0, T0, R1, T1, alpha)
        depth = depth_min + alpha * (depth_max - depth_min)

        pts_cam = np.linalg.inv(K) @ np.vstack((x_coords.ravel(), y_coords.ravel(), np.ones(w*h)))
        pts_cam *= depth
        pts_cam = pts_cam.T

        pts_world = (np.linalg.inv(Rv) @ (pts_cam.T - Tv.reshape(3, 1))).T
        proj1 = project_points(pts_world, K, R0, T0)
        proj2 = project_points(pts_world, K, R1, T1)

        for idx, (x0, y0, p1x, p1y, p2x, p2y) in enumerate(zip(
            x_coords.ravel(), y_coords.ravel(), proj1[:, 0], proj1[:, 1], proj2[:, 0], proj2[:, 1]
        )):
            scores = []
            valid = 0

            # Try NCC from img1 -> img2
            x0i, y0i = int(x0), int(y0)
            p2xi, p2yi = int(round(p2x)), int(round(p2y))
            if (x0i - half_patch >= 0 and x0i + half_patch < w and y0i - half_patch >= 0 and y0i + half_patch < h and
                p2xi - half_patch >= 0 and p2xi + half_patch < w and p2yi - half_patch >= 0 and p2yi + half_patch < h):
                patch1 = img1[y0i - half_patch:y0i + half_patch + 1, x0i - half_patch:x0i + half_patch + 1].astype(np.float32)
                patch2 = img2[p2yi - half_patch:p2yi + half_patch + 1, p2xi - half_patch:p2xi + half_patch + 1].astype(np.float32)
                scores.append(normalized_cross_correlation(patch1, patch2))
                valid += 1

            # Try NCC from img2 -> img1
            p1xi, p1yi = int(round(p1x)), int(round(p1y))
            if (p1xi - half_patch >= 0 and p1xi + half_patch < w and p1yi - half_patch >= 0 and p1yi + half_patch < h and
                x0i - half_patch >= 0 and x0i + half_patch < w and y0i - half_patch >= 0 and y0i + half_patch < h):
                patch1 = img2[p1yi - half_patch:p1yi + half_patch + 1, p1xi - half_patch:p1xi + half_patch + 1].astype(np.float32)
                patch2 = img1[y0i - half_patch:y0i + half_patch + 1, x0i - half_patch:x0i + half_patch + 1].astype(np.float32)
                scores.append(normalized_cross_correlation(patch1, patch2))
                valid += 1

            if valid > 0:
                cost = np.mean(scores)
                if cost > cost_volume[y0i, x0i]:
                    cost_volume[y0i, x0i] = cost
                    depth_map[y0i, x0i] = depth

        print(f"Depth step {i+1}/{depth_steps}, alpha={alpha:.2f} completed")

    return depth_map, cost_volume


def bilinear_sample(img, x, y):
    """Perform bilinear sampling on image `img` at (x, y) coords (float arrays)
    
    :param img: Input image (H, W, C) or (H, W) for grayscale.
    :param x: x-coordinates (float array).
    :param y: y-coordinates (float array).
    :return: Sampled pixel values at (x, y) as uint8 array.
    """

    h, w = img.shape[:2]

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Clamp values to valid indices
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    # Get pixel values
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    # Compute weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia.T * wa + Ib.T * wb + Ic.T * wc + Id.T * wd).T.astype(np.uint8)


def colorize_depth_map_with_projection(depth_map, K, R1, T1, R2, T2, img2Color):
    """
    Colorize the depth map by projecting 3D points into the second camera
    and sampling colors from img2Color using bilinear interpolation.

    :param depth_map: Depth map from plane sweep stereo (H, W).
    :param K: Camera intrinsic matrix (3x3).
    :param R1: Rotation matrix of the first camera pose (3x3).  
    :param T1: Translation vector of the first camera pose (3,).
    :param R2: Rotation matrix of the second camera pose (3x3).
    :param T2: Translation vector of the second camera pose (3,).
    :param img2Color: Color image corresponding to the second camera (H, W, 3).
    :return: Colorized depth map (H, W, 3) where each pixel color corresponds to the projected depth.
    """

    h, w = depth_map.shape
    output_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Create pixel grid in camera 1 image
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Flatten pixel coordinates and corresponding depths
    pixels_hom = np.vstack((x_coords.ravel(), y_coords.ravel(), np.ones(w * h))) 
    depths = depth_map.ravel()

    # Filter out invalid depths
    valid_mask = depths > 0
    valid_pixels_hom = pixels_hom[:, valid_mask]
    valid_depths = depths[valid_mask]

    # Backproject valid pixels to camera 1 coordinates
    pts_cam1 = np.linalg.inv(K) @ valid_pixels_hom  
    pts_cam1 *= valid_depths
    pts_cam1 = pts_cam1.T 

    # Transform points from camera 1 to world coordinates (R1, T1)
    pts_world = (R1 @ pts_cam1.T + T1.reshape(3, 1)).T

    # Transform points from world to camera 2 coordinates
    pts_cam2 = (R2.T @ (pts_world - T2.reshape(1, 3)).T).T 

    # Project points into camera 2 image plane
    pts_proj_cam2 = (K @ pts_cam2.T).T  # Nx3
    pts_proj_cam2 = pts_proj_cam2[:, :2] / pts_proj_cam2[:, 2:]

    x2f = pts_proj_cam2[:, 0]
    y2f = pts_proj_cam2[:, 1]

    h2, w2 = img2Color.shape[:2]
    valid_proj_mask = (x2f >= 0) & (x2f < w2 - 1) & (y2f >= 0) & (y2f < h2 - 1)

    # Only sample colors where projections are valid
    x2_valid = x2f[valid_proj_mask]
    y2_valid = y2f[valid_proj_mask]

    sampled_colors = bilinear_sample(img2Color, x2_valid, y2_valid)

    indices_all = np.nonzero(valid_mask)[0]       
    indices_valid = indices_all[valid_proj_mask]  

    output_img_flat = output_img.reshape(-1, 3)
    output_img_flat[indices_valid] = sampled_colors

    return output_img



def create_camera_mesh(K, R, T, scale=0.2, color=[1, 0, 0]):
    """Create a small pyramid mesh to represent a camera pose

    :param K: Camera intrinsic matrix (3x3).
    :param R: Rotation matrix (3x3) representing camera orientation.
    :param T: Translation vector (3,) representing camera position. 
    :param scale: Scale factor for the camera mesh.
    :param color: Color of the camera mesh as RGB list.
    :return: Open3D TriangleMesh representing the camera pose. 
    """
    
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T.flatten()
    mesh.transform(transform)
    mesh.paint_uniform_color(color)
    return mesh


def create_virtual_plane(depth, K, R, T, size=(2.0, 1.5), density=20):
    """
    Create a virtual plane mesh at given depth in camera coords.
    The plane is placed perpendicular to camera z-axis.

    :param depth: Depth at which to place the plane.
    :param K: Camera intrinsic matrix (3x3).
    :param R: Rotation matrix (3x3) representing camera orientation.
    :param T: Translation vector (3,) representing camera position.
    :param size: Size of the plane in world coordinates (width, height).
    :param density: Number of points along each axis to create the grid.
    :return: Open3D PointCloud representing the virtual plane.
    """

    width, height = size
    xs = np.linspace(-width / 2, width / 2, density)
    ys = np.linspace(-height / 2, height / 2, density)
    xs, ys = np.meshgrid(xs, ys)
    zs = np.ones_like(xs) * depth

    points_cam = np.vstack((xs.flatten(), ys.flatten(), zs.flatten())).T  # Nx3

    points_world = (R @ points_cam.T + T.reshape(3, 1)).T

    # Create mesh from points
    plane = o3d.geometry.PointCloud()
    plane.points = o3d.utility.Vector3dVector(points_world)

    return plane


def visualize_cameras_and_plane(K, R0, T0, R1, T1, Rv, Tv, depth_plane):
    """
    Visualize two camera poses and a virtual camera pose with a plane at a specified depth.

    :param K: Camera intrinsic matrix (3x3).
    :param R0: Rotation matrix of the first camera pose (3x3).
    :param T0: Translation vector of the first camera pose (3,).
    :param R1: Rotation matrix of the second camera pose (3x3).
    :param T1: Translation vector of the second camera pose (3,).
    :param Rv: Rotation matrix of the virtual camera pose (3x3).
    :param Tv: Translation vector of the virtual camera pose (3,).
    :param depth_plane: Depth at which to place the virtual plane.
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    cam0_mesh = create_camera_mesh(K, R0, T0, scale=0.1, color=[1, 0, 0])  
    cam1_mesh = create_camera_mesh(K, R1, T1, scale=0.1, color=[0, 1, 0])  
    camv_mesh = create_camera_mesh(K, Rv, Tv, scale=0.1, color=[0, 0, 1])  

    plane_mesh = create_virtual_plane(depth_plane, K, Rv, Tv, size=(2.0, 1.5), density=20)
    plane_mesh.paint_uniform_color([0.5, 0.5, 0.8])

    vis.add_geometry(cam0_mesh)
    vis.add_geometry(cam1_mesh)
    vis.add_geometry(camv_mesh)
    vis.add_geometry(plane_mesh)

    vis.capture_screen_image("camera_plane_visualization.png")

    vis.run()
    vis.destroy_window()


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

    # Decode Gray codes
    result0 = decode_gray_pattern(images0)
    result1 = decode_gray_pattern(images1)

    # Load color images
    img1Color = cv2.resize(cv2.imread(images_view0[0], cv2.IMREAD_COLOR), (1920, 1080))
    img2Color = cv2.resize(cv2.imread(images_view1[0], cv2.IMREAD_COLOR), (1920, 1080))

    keypoints1, keypoints2, matches = find_correspondences(result0, result1)

    K = np.array([
        [9.51663140e+03, 0.00000000e+00, 2.81762458e+03],
        [0.00000000e+00, 8.86527952e+03, 1.14812762e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    E, mask, pts1, pts2 = compute_essential_matrix(keypoints1, keypoints2, matches, K)

    _, R, T, _ = cv2.recoverPose(E, pts1, pts2, K)

    R1 = np.eye(3)       
    T1 = np.zeros((3,1))

    R2 = R             
    T2 = T


    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  
    P2 = K @ np.hstack((R, T))                         

    print("Camera 1 pose:\nR:", R1, "\nT:", T1)
    print("Camera 2 pose:\nR:", R2, "\nT:", T2)

    img1Gray = cv2.cvtColor(img1Color, cv2.COLOR_BGR2GRAY)
    img2Gray = cv2.cvtColor(img2Color, cv2.COLOR_BGR2GRAY)

    depth_map, cost_vol = plane_sweep_depth(img1Gray, img2Gray, K, R1, T1, R2, T2,
                                            depth_min=0.5, depth_max=5.0, depth_steps=50, patch_size=7)

    # Colorize depth map using projection into second camera
    colorized_depth = colorize_depth_map_with_projection(depth_map, K, R1, T1, R2, T2, img2Color)

    cv2.imwrite("depth_map2.png", (depth_map / depth_map.max() * 255).astype(np.uint8))
    cv2.imwrite("colorized_depth2.png", colorized_depth)

    cv2.imshow("Depth Map", (depth_map / depth_map.max() * 255).astype(np.uint8))
    cv2.imshow("Colorized Depth Map", colorized_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create mask of invalid (black) pixels
    mask = np.all(colorized_depth == 0, axis=2).astype(np.uint8) * 255

    # Inpaint using surrounding valid color
    inpainted = cv2.inpaint(colorized_depth, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    cv2.imwrite("colorized_depth_inpainted.png", inpainted)
    cv2.imshow("Inpainted Colorized Depth", inpainted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    alpha_virtual = 0.5
    Rv, Tv = interpolate_pose(R1, T1, R2, T2, alpha_virtual)
    depth_virtual_plane = 0.5 + alpha_virtual * (5.0 - 0.5) 

    visualize_cameras_and_plane(K, R1, T1, R2, T2, Rv, Tv, depth_virtual_plane)