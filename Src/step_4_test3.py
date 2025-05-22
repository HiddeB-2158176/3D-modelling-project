import numpy as np
import cv2
import glob

from step_2 import decode_gray_pattern, find_correspondences
from step_3 import compute_essential_matrix


def average_rotations(R1, R2):
    """Average two rotation matrices using SVD to ensure a valid rotation matrix."""
    R = R1 + R2
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def get_virtual_camera_pose(P1, P2):
    """
    Calculate the virtual camera pose as the midpoint between P1 and P2.
    Properly average rotation matrices using SVD.
    Returns 4x4 pose matrix.
    """
    R1, t1 = P1[:, :3], P1[:, 3:]
    R2, t2 = P2[:, :3], P2[:, 3:]

    R_virtual = average_rotations(R1, R2)
    t_virtual = (t1 + t2) / 2.0

    pose_virtual = np.eye(4)
    pose_virtual[:3, :3] = R_virtual
    pose_virtual[:3, 3] = t_virtual.flatten()
    return pose_virtual


def deproject_corners(K, pose, depth, w, h):
    # corners in virtual image pixel coords
    corners_2d = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    # Convert pixel corners to normalized coordinates
    corners_norm = cv2.undistortPoints(corners_2d.reshape(-1, 1, 2), K, None).reshape(-1, 2)

    # Backproject normalized coords to 3D points at given depth in virtual camera frame
    corners_cam = np.hstack([corners_norm * depth, np.full((4, 1), depth)])

    # Convert to homogeneous coordinates (4x4)
    corners_cam_h = np.hstack([corners_cam, np.ones((4, 1))]).T

    # Transform to world coordinates (invert virtual pose)
    pose_inv = np.linalg.inv(pose)
    corners_world = pose_inv @ corners_cam_h  # shape (4,4)

    return corners_world[:3].T  # (4x3)


def project_points(K, pose, points_3d):
    """
    Project 3D points (Nx3) to image plane of camera defined by pose.
    """
    points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))]).T  # 4xN
    proj = K @ (pose[:3, :] @ points_3d_h)  # 3xN
    proj = proj[:2] / proj[2]
    return proj.T  # Nx2


def build_pose(R, t):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.flatten()
    return pose


def generate_virtual_view(img1, img2, K, P1, P2, num_planes=200):
    h, w = img1.shape[:2]

    R1, t1 = P1[:, :3], P1[:, 3:]
    R2, t2 = P2[:, :3], P2[:, 3:]

    pose1 = build_pose(R1, t1)
    pose2 = build_pose(R2, t2)
    pose_virtual = get_virtual_camera_pose(P1, P2)

    # Use grayscale for cost
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    best_img = np.zeros((h, w, 3), dtype=np.float32)
    best_score = np.full((h, w), np.inf, dtype=np.float32)

    # Depth range: You might want to adjust this for your scene scale
    depth_min = 0.1
    depth_max = 1000

    virtual_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    for depth in np.linspace(depth_min, depth_max, num_planes):
        # 1) Deproject virtual image corners to 3D world coords at depth plane
        corners_3d = deproject_corners(K, pose_virtual, depth, w, h)

        # 2) Project corners into input cameras
        proj1 = project_points(K, pose1, corners_3d)
        proj2 = project_points(K, pose2, corners_3d)

        # 3) Check projections inside input images
        if (np.any(proj1 < 0) or np.any(proj1[:, 0] >= w) or np.any(proj1[:, 1] >= h) or
                np.any(proj2 < 0) or np.any(proj2[:, 0] >= w) or np.any(proj2[:, 1] >= h)):
            continue

        # 4) Compute homographies from input images to virtual view
        # Homography: input image points → virtual view points
        H1, status1 = cv2.findHomography(proj1, virtual_corners)
        H2, status2 = cv2.findHomography(proj2, virtual_corners)
        if H1 is None or H2 is None:
            continue

        # 5) Warp input images to virtual view
        img1_warped = cv2.warpPerspective(gray1, H1, (w, h))
        img2_warped = cv2.warpPerspective(gray2, H2, (w, h))

        # 6) Compute absolute difference cost (can also try squared diff)
        cost = np.abs(img1_warped - img2_warped)

        # Optional: threshold cost to ignore bad matches
        max_cost_thresh = 50.0
        cost_mask = cost < max_cost_thresh

        # 7) Update best scores and best image pixels where cost improves
        update_mask = (cost < best_score) & cost_mask

        best_score[update_mask] = cost[update_mask]
        # For color image, warp original images with same homographies
        img1_color_warped = cv2.warpPerspective(img1, H1, (w, h))
        img2_color_warped = cv2.warpPerspective(img2, H2, (w, h))
        best_img[update_mask] = 0.5 * (img1_color_warped[update_mask] + img2_color_warped[update_mask])

    valid_pixel_count = np.count_nonzero(best_score < np.inf)
    print(f"Valid pixels in final virtual view: {valid_pixel_count} out of {w*h}")

    if valid_pixel_count == 0:
        print("Warning: No valid pixels found! Try adjusting depth range or camera poses.")

    return np.clip(best_img, 0, 255).astype(np.uint8)


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

    # Camera intrinsic matrix K
    K = np.array([
        [9.51663140e+03, 0.00000000e+00, 2.81762458e+03],
        [0.00000000e+00, 8.86527952e+03, 1.14812762e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    # Compute essential matrix, mask, matched points
    E, mask, pts1, pts2 = compute_essential_matrix(keypoints1, keypoints2, matches, K)

    # Recover pose
    _, R, T, _ = cv2.recoverPose(E, pts1, pts2, K)

    # Projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1 at origin
    P2 = K @ np.hstack((R, T))                         # Camera 2 pose

    # Generate virtual view with plane sweep stereo
    virtual_view = generate_virtual_view(img1Color, img2Color, K, P1, P2, num_planes=50)

    cv2.imwrite("virtual_view.png", virtual_view)
    cv2.imshow("Virtual View", virtual_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
