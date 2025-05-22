import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp
import glob

from step_2 import decode_gray_pattern, find_correspondences
from step_3 import compute_essential_matrix


def interpolate_pose(R0, T0, R1, T1, alpha):
    times = [0, 1]
    rotations = SciRotation.from_matrix([R0, R1])
    slerp = Slerp(times, rotations)
    r_interp = slerp([alpha])[0]
    T_interp = (1 - alpha) * T0 + alpha * T1
    return r_interp.as_matrix(), T_interp


def project_points(points_3d, K, R, T):
    """Project 3D points into 2D using camera intrinsics and extrinsics"""
    P = K @ np.hstack((R, T.reshape(3, 1)))  # 3x4 projection matrix
    points_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T  # 4xN
    pts_2d_hom = P @ points_hom  # 3xN
    pts_2d = (pts_2d_hom[:2] / pts_2d_hom[2]).T  # Nx2 normalized pixel coords
    return pts_2d


def normalized_cross_correlation(patch1, patch2, epsilon=1e-5):
    """Compute normalized cross-correlation between two image patches."""
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    numerator = np.sum((patch1 - mean1) * (patch2 - mean2))
    denominator = np.sqrt(np.sum((patch1 - mean1) ** 2) * np.sum((patch2 - mean2) ** 2)) + epsilon
    return numerator / denominator

def match_keypoints(img1, img2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Use SIFT detector
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match descriptors using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    points1 = []
    points2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            points1.append(kp1[m.queryIdx].pt)
            points2.append(kp2[m.trainIdx].pt)

    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)

    return points1, points2, good_matches


def plane_sweep_stereo_two_view(ref_img, src_img, K, depth_range, num_planes):
    height, width = ref_img.shape[:2]

    # Step 1: Match keypoints and estimate relative pose
    kp1, kp2, matches = match_keypoints(ref_img, src_img)
    if len(matches) < 8:
        raise ValueError("Not enough matches to estimate pose")

    E, mask = cv2.findEssentialMat(kp1, kp2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, kp1, kp2, K)

    # Step 2: Triangulate 3D keypoints and estimate scene center
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    points_4d = cv2.triangulatePoints(P1, P2, kp1.T, kp2.T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    scene_center = np.mean(points_3d, axis=0)
    ref_center_cam = scene_center  # Since ref is identity pose
    center_depth = ref_center_cam[2]

    # Step 3: Define depth planes around scene center
    min_depth = max(depth_range[0], center_depth - 0.5)
    max_depth = min(depth_range[1], center_depth + 0.5)
    depth_planes = np.linspace(min_depth, max_depth, num_planes)

    # Step 4: Prepare projection helpers
    K_inv = np.linalg.inv(K)
    yv, xv = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    pixels_h = np.stack([xv, yv, np.ones_like(xv)], axis=-1).reshape(-1, 3).T

    cost_volume = np.zeros((num_planes, height, width))

    for d_idx, depth in enumerate(depth_planes):
        # Step 5: Compute homography from ref to src for plane at depth
        n = np.array([0, 0, 1]).reshape(3, 1)
        d = depth
        t_rel = t.reshape(3, 1)
        H = K @ (R - (t_rel @ n.T) / d) @ K_inv

        # Step 6: Warp source image to reference view
        warped_src = cv2.warpPerspective(src_img, H, (width, height))

        # Step 7: Compute cost map
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        warped_gray = cv2.cvtColor(warped_src, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cost_map = np.abs(ref_gray - warped_gray)

        cost_volume[d_idx] = cost_map

    # Step 8: Select depth with lowest cost
    depth_indices = np.argmin(cost_volume, axis=0)
    depth_map = depth_planes[depth_indices]

    return depth_map



def colorize_depth_map_with_projection(depth_map, K, R1, T1, R2, T2, img2Color):
    """
    Colorize the depth map by projecting 3D points into the second camera
    and sampling colors from img2Color.
    """

    h, w = depth_map.shape
    output_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Create pixel grid in camera 1 image
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Flatten pixel coordinates and corresponding depths
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

    # Transform points from camera 1 to world coordinates (cam1 at origin)
    pts_world = (R1 @ pts_cam1.T + T1.reshape(3, 1)).T  # Nx3

    # Transform points from world to camera 2 coordinates
    pts_cam2 = (R2.T @ (pts_world - T2.reshape(1, 3)).T).T  # Nx3

    # Project points into camera 2 image plane
    pts_proj_cam2 = (K @ pts_cam2.T).T  # Nx3
    pts_proj_cam2 = pts_proj_cam2[:, :2] / pts_proj_cam2[:, 2:]

    # Round and convert to int pixel coordinates
    x2 = np.round(pts_proj_cam2[:, 0]).astype(int)
    y2 = np.round(pts_proj_cam2[:, 1]).astype(int)

    # Clamp projected points inside img2 bounds
    h2, w2 = img2Color.shape[:2]
    valid_proj_mask = (x2 >= 0) & (x2 < w2) & (y2 >= 0) & (y2 < h2)

    # Prepare flat output image for assignment
    output_img_flat = output_img.reshape(-1, 3)

    # Assign colors from img2Color to output image where valid
    indices_1 = np.nonzero(valid_mask)[0]  # indices of valid depths in flattened array
    indices_2 = indices_1[valid_proj_mask]  # indices where projection is valid

    output_img_flat[indices_2] = img2Color[y2[valid_proj_mask], x2[valid_proj_mask]]

    return output_img


def create_camera_mesh(K, R, T, scale=0.2, color=[1, 0, 0]):
    """Create a small pyramid mesh to represent a camera pose"""
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
    """
    width, height = size
    # Create grid points in camera frame at depth
    xs = np.linspace(-width / 2, width / 2, density)
    ys = np.linspace(-height / 2, height / 2, density)
    xs, ys = np.meshgrid(xs, ys)
    zs = np.ones_like(xs) * depth

    points_cam = np.vstack((xs.flatten(), ys.flatten(), zs.flatten())).T  # Nx3

    # Transform points to world
    points_world = (R @ points_cam.T + T.reshape(3, 1)).T

    # Create mesh from points
    plane = o3d.geometry.PointCloud()
    plane.points = o3d.utility.Vector3dVector(points_world)

    return plane


def visualize_cameras_and_plane(K, R0, T0, R1, T1, Rv, Tv, depth_plane):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    cam0_mesh = create_camera_mesh(K, R0, T0, scale=0.1, color=[1, 0, 0])  # Red
    cam1_mesh = create_camera_mesh(K, R1, T1, scale=0.1, color=[0, 1, 0])  # Green
    camv_mesh = create_camera_mesh(K, Rv, Tv, scale=0.1, color=[0, 0, 1])  # Blue virtual cam

    plane_mesh = create_virtual_plane(depth_plane, K, Rv, Tv, size=(2.0, 1.5), density=20)
    plane_mesh.paint_uniform_color([0.5, 0.5, 0.8])

    vis.add_geometry(cam0_mesh)
    vis.add_geometry(cam1_mesh)
    vis.add_geometry(camv_mesh)
    vis.add_geometry(plane_mesh)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # Load grayscale images for decoding
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

    # Load color images for final visualization
    img1Color = cv2.resize(cv2.imread(images_view0[0], cv2.IMREAD_COLOR), (1920, 1080))
    img2Color = cv2.resize(cv2.imread(images_view1[0], cv2.IMREAD_COLOR), (1920, 1080))

    # Find correspondences (keypoints1, keypoints2, matches)
    keypoints1, keypoints2, matches = find_correspondences(result0, result1)

    K = np.array([
        [9.51663140e+03, 0.00000000e+00, 2.81762458e+03],
        [0.00000000e+00, 8.86527952e+03, 1.14812762e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    # Compute essential matrix, mask, matched points
    E, mask, pts1, pts2 = compute_essential_matrix(keypoints1, keypoints2, matches, K)

    # Recover pose
    _, R, T, _ = cv2.recoverPose(E, pts1, pts2, K)

    R1 = np.eye(3)       # Camera 1 at origin
    T1 = np.zeros((3,1))

    R2 = R               # From recoverPose
    T2 = T

    # Projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1 at origin
    P2 = K @ np.hstack((R, T))                         # Camera 2 pose

    print("Camera 1 pose:\nR:", R1, "\nT:", T1)
    print("Camera 2 pose:\nR:", R2, "\nT:", T2)

    # Convert to grayscale for plane sweep
    img1Gray = cv2.cvtColor(img1Color, cv2.COLOR_BGR2GRAY)
    img2Gray = cv2.cvtColor(img2Color, cv2.COLOR_BGR2GRAY)

    # Compute depth map by plane sweep stereo
    depth_map = plane_sweep_stereo_two_view(img1Color, img2Color, K, depth_range=(1.0, 10.0), num_planes=32)


    # Colorize depth map using projection into second camera
    colorized_depth = colorize_depth_map_with_projection(depth_map, K, R1, T1, R2, T2, img2Color)

    # Save and show results
    cv2.imwrite("depth_map3.png", (depth_map / depth_map.max() * 255).astype(np.uint8))
    cv2.imwrite("colorized_depth3.png", colorized_depth)

    cv2.imshow("Depth Map", (depth_map / depth_map.max() * 255).astype(np.uint8))
    cv2.imshow("Colorized Depth Map", colorized_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Visualize cameras and a virtual camera pose + plane at mid-depth
    alpha_virtual = 0.5
    Rv, Tv = interpolate_pose(R1, T1, R2, T2, alpha_virtual)
    depth_virtual_plane = 0.5 + alpha_virtual * (5.0 - 0.5)  # Mid-depth

    visualize_cameras_and_plane(K, R1, T1, R2, T2, Rv, Tv, depth_virtual_plane) 