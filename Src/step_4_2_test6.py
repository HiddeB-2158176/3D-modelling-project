import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp
import glob

from step_2 import decode_gray_pattern, find_correspondences
from step_3 import compute_essential_matrix


# This file contains a similar implementation to step_4_2_test5.py, but with a different approach to
# plane sweep stereo using two views and a more structured way to visualize the results.
# This file generates the best results we came up with, as is seen in the Result folder.

def interpolate_pose(R0, T0, R1, T1, alpha):
    """
    Interpolate between two camera poses using spherical linear interpolation (SLERP).

    :param R0: Rotation matrix of the first camera pose (3x3).
    :param T0: Translation vector of the first camera pose (3x1).
    :param R1: Rotation matrix of the second camera pose (3x3).
    :param T1: Translation vector of the second camera pose (3x1).
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
    """Project 3D points into 2D using camera intrinsics and extrinsics

    :param points_3d: Nx3 array of 3D points in world coordinates.
    :param K: Camera intrinsic matrix (3x3).
    :param R: Rotation matrix (3x3) from world to camera coordinates.
    :param T: Translation vector (3x1) from world to camera coordinates.
    :return: Nx2 array of 2D points in normalized pixel coordinates.
    """

    P = K @ np.hstack((R, T.reshape(3, 1))) 
    points_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T 
    pts_2d_hom = P @ points_hom 
    pts_2d = (pts_2d_hom[:2] / pts_2d_hom[2]).T 

    return pts_2d


def normalized_cross_correlation(patch1, patch2, epsilon=1e-5):
    """Compute normalized cross-correlation between two image patches.

    :param patch1: First image patch (H x W).
    :param patch2: Second image patch (H x W).
    :param epsilon: Small value to avoid division by zero.
    :return: Normalized cross-correlation value.
    """

    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    numerator = np.sum((patch1 - mean1) * (patch2 - mean2))
    denominator = np.sqrt(np.sum((patch1 - mean1) ** 2) * np.sum((patch2 - mean2) ** 2)) + epsilon
    return numerator / denominator


def plane_sweep_depth(img1, img2, K, R0, T0, R1, T1,
                      depth_min=0.5, depth_max=5.0, depth_steps=50, patch_size=7):
    """
    Compute depth map using plane sweep between two views.

    :param img1: First grayscale image (H x W).
    :param img2: Second grayscale image (H x W).
    :param K: Camera intrinsic matrix (3x3).
    :param R0: Rotation matrix of the first camera pose (3x3).
    :param T0: Translation vector of the first camera pose (3x1).
    :param R1: Rotation matrix of the second camera pose (3x3).
    :param T1: Translation vector of the second camera pose (3x1).
    :param depth_min: Minimum depth value to consider.
    :param depth_max: Maximum depth value to consider.
    :param depth_steps: Number of depth steps to sample.
    :param patch_size: Size of the square patches to compare (must be odd).
    :return: Depth map (H x W) and cost volume (H x W).
    """

    h, w = img1.shape[:2]
    depth_map = np.zeros((h, w), dtype=np.float32)
    cost_volume = np.full((h, w), -np.inf, dtype=np.float32)

    half_patch = patch_size // 2
    alphas = np.linspace(0, 1, depth_steps)

    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    for i, alpha in enumerate(alphas):
        R_virtual, T_virtual = interpolate_pose(R0, T0, R1, T1, alpha)

        # Compute current depth plane in virtual view 
        depth = depth_min + alpha * (depth_max - depth_min)

        # Backproject pixels to 3D points on current depth plane (virtual camera coords)
        pts_cam = np.linalg.inv(K) @ np.vstack((x_coords.ravel(), y_coords.ravel(), np.ones(w * h))) 
        pts_cam *= depth  
        pts_cam = pts_cam.T 

        pts_world = (np.linalg.inv(R_virtual) @ (pts_cam.T - T_virtual.reshape(3, 1))).T

        proj1 = project_points(pts_world, K, R0, T0) 
        proj2 = project_points(pts_world, K, R1, T1)  

        # Evaluate cost (NCC) only for valid projections with patch extraction
        for idx, (x0, y0, x1, y1) in enumerate(zip(x_coords.ravel(), y_coords.ravel(), proj2[:, 0], proj2[:, 1])):
            x1_int, y1_int = int(round(x1)), int(round(y1))

            # Check bounds for patches in both images
            if (x0 - half_patch < 0 or x0 + half_patch >= w or y0 - half_patch < 0 or y0 + half_patch >= h or
                    x1_int - half_patch < 0 or x1_int + half_patch >= w or y1_int - half_patch < 0 or y1_int + half_patch >= h):
                continue

            patch1 = img1[y0 - half_patch:y0 + half_patch + 1, x0 - half_patch:x0 + half_patch + 1].astype(np.float32)
            patch2 = img2[y1_int - half_patch:y1_int + half_patch + 1, x1_int - half_patch:x1_int + half_patch + 1].astype(np.float32)

            cost = normalized_cross_correlation(patch1, patch2)

            # Update depth map if this cost is better
            if cost > cost_volume[y0, x0]:
                cost_volume[y0, x0] = cost
                depth_map[y0, x0] = depth

        print(f"Depth step {i + 1}/{depth_steps}, alpha={alpha:.2f} completed")

    return depth_map, cost_volume


def colorize_depth_map_with_projection(depth_map, K, R1, T1, R2, T2, img2Color):
    """
    Colorize the depth map by projecting 3D points into the second camera
    and sampling colors from img2Color.

    :param depth_map: Depth map from plane sweep (H x W).
    :param K: Camera intrinsic matrix (3x3).
    :param R1: Rotation matrix of the first camera pose (3x3).
    :param T1: Translation vector of the first camera pose (3x1).
    :param R2: Rotation matrix of the second camera pose (3x3).
    :param T2: Translation vector of the second camera pose (3x1).
    :param img2Color: Color image from the second camera (H x W x 3).
    :return: Colorized depth map (H x W x 3).
    """

    h, w = depth_map.shape
    output_img = np.zeros((h, w, 3), dtype=np.uint8)

    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    pixels_hom = np.vstack((x_coords.ravel(), y_coords.ravel(), np.ones(w * h)))  # 3xN
    depths = depth_map.ravel()

    # Filter out invalid depths
    valid_mask = depths > 0
    valid_pixels_hom = pixels_hom[:, valid_mask]
    valid_depths = depths[valid_mask]

    # Backproject valid pixels to camera 1 coordinates
    pts_cam1 = np.linalg.inv(K) @ valid_pixels_hom  # 3xN
    pts_cam1 *= valid_depths  # scale by depth
    pts_cam1 = pts_cam1.T  # Nx3

    pts_world = (R1 @ pts_cam1.T + T1.reshape(3, 1)).T  # Nx3

    pts_cam2 = (R2.T @ (pts_world - T2.reshape(1, 3)).T).T  # Nx3

    # Project points into camera 2 image plane
    pts_proj_cam2 = (K @ pts_cam2.T).T  # Nx3
    pts_proj_cam2 = pts_proj_cam2[:, :2] / pts_proj_cam2[:, 2:]

    x2 = np.round(pts_proj_cam2[:, 0]).astype(int)
    y2 = np.round(pts_proj_cam2[:, 1]).astype(int)

    # Clamp projected points inside img2 bounds
    h2, w2 = img2Color.shape[:2]
    valid_proj_mask = (x2 >= 0) & (x2 < w2) & (y2 >= 0) & (y2 < h2)

    output_img_flat = output_img.reshape(-1, 3)

    # Assign colors from img2Color to output image where valid
    indices_1 = np.nonzero(valid_mask)[0]  
    indices_2 = indices_1[valid_proj_mask]  

    output_img_flat[indices_2] = img2Color[y2[valid_proj_mask], x2[valid_proj_mask]]

    return output_img


def create_camera_mesh(K, R, T, scale=0.2, color=[1, 0, 0]):
    """Create a small pyramid mesh to represent a camera pose
    
    :param K: Camera intrinsic matrix (3x3).
    :param R: Rotation matrix (3x3) from world to camera coordinates.
    :param T: Translation vector (3x1) from world to camera coordinates.
    :param scale: Scale factor for the camera mesh.
    :param color: Color of the camera mesh in RGB format.
    :return: Open3D TriangleMesh representing the camera.
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
    :param R: Rotation matrix (3x3) from world to camera coordinates.
    :param T: Translation vector (3x1) from world to camera coordinates.
    :param size: Size of the plane in world coordinates (width, height).
    :param density: Number of points along each axis to sample.
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
    :param T0: Translation vector of the first camera pose (3x1).
    :param R1: Rotation matrix of the second camera pose (3x3).
    :param T1: Translation vector of the second camera pose (3x1).
    :param Rv: Rotation matrix of the virtual camera pose (3x3).
    :param Tv: Translation vector of the virtual camera pose (3x1).
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

    result0 = decode_gray_pattern(images0)
    result1 = decode_gray_pattern(images1)

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

    cv2.imwrite("depth_map3.png", (depth_map / depth_map.max() * 255).astype(np.uint8))
    cv2.imwrite("colorized_depth3.png", colorized_depth)

    cv2.imshow("Depth Map", (depth_map / depth_map.max() * 255).astype(np.uint8))
    cv2.imshow("Colorized Depth Map", colorized_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create mask of invalid (black) pixels
    mask = np.all(colorized_depth == 0, axis=2).astype(np.uint8) * 255

    # Inpaint using surrounding valid color
    inpainted = cv2.inpaint(colorized_depth, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    cv2.imwrite("colorized_depth_inpainted3.png", inpainted)
    cv2.imshow("Inpainted Colorized Depth", inpainted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    alpha_virtual = 0.5
    Rv, Tv = interpolate_pose(R1, T1, R2, T2, alpha_virtual)
    depth_virtual_plane = 0.5 + alpha_virtual * (5.0 - 0.5)  

    visualize_cameras_and_plane(K, R1, T1, R2, T2, Rv, Tv, depth_virtual_plane)
