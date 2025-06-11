import numpy as np
import cv2 as cv
import glob

# Termination criteria for cornerSubPix and solvePnP
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def get_3d_points_from_camera_view_of_projected_pattern(img_path, K_cam, D_cam,
                                                       physical_chessboard_objp_3D, physical_board_dims,
                                                       projected_pattern_dims):
    """
    Processes a camera image of a physical chessboard with a projected pattern.
    Calculates 3D world coordinates of projected corners.
    """
    img = cv.imread(img_path)
    if img is None:
        print(f"Error: Image not loaded from {img_path}")
        return None, None
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 1. Find physical chessboard
    ret_phys, corners_phys_2D = cv.findChessboardCorners(gray, physical_board_dims, None)
    if not ret_phys:
        print(f"Physical chessboard not found in {img_path}")
        return None, None

    corners_phys_2D_refined = cv.cornerSubPix(gray, corners_phys_2D, (11, 11), (-1, -1), criteria)
    ret_pose, rvec_phys_world_to_cam, tvec_phys_world_to_cam = cv.solvePnP(
        physical_chessboard_objp_3D, corners_phys_2D_refined, K_cam, D_cam
    )
    if not ret_pose:
        print(f"Could not solvePnP for physical chessboard in {img_path}")
        return None, None
    R_phys_world_to_cam, _ = cv.Rodrigues(rvec_phys_world_to_cam)

    # 2. Detect projected pattern corners in camera image
    ret_proj, corners_proj_cam_2D = cv.findChessboardCorners(gray, projected_pattern_dims, None)
    if not ret_proj:
        print(f"Projected pattern not found in {img_path}")
        return None, None
    
    corners_proj_cam_2D_refined = cv.cornerSubPix(gray, corners_proj_cam_2D, (11, 11), (-1, -1), criteria)
    corners_proj_cam_2D_undistorted = cv.undistortPoints(corners_proj_cam_2D_refined, K_cam, D_cam, P=K_cam)
    corners_proj_cam_2D_undistorted = corners_proj_cam_2D_undistorted.reshape(-1, 2)

    # 3. Back-project camera 2D points to 3D points on physical chessboard plane
    object_points_for_projector_3D_world = []
    K_cam_inv = np.linalg.inv(K_cam)
    N_cam = R_phys_world_to_cam[:, 2] 
    
    for pt_cam_undistorted_px in corners_proj_cam_2D_undistorted:
        ray_cam_norm = K_cam_inv @ np.array([pt_cam_undistorted_px[0], pt_cam_undistorted_px[1], 1.0])
        denominator = N_cam @ ray_cam_norm
        if np.abs(denominator) < 1e-6:
            continue
        
        s = (N_cam @ tvec_phys_world_to_cam.flatten()) / denominator
        P_cam_3D = s * ray_cam_norm
        P_world_3D = R_phys_world_to_cam.T @ (P_cam_3D - tvec_phys_world_to_cam.flatten())
        object_points_for_projector_3D_world.append(P_world_3D)

    if len(object_points_for_projector_3D_world) != (projected_pattern_dims[0] * projected_pattern_dims[1]):
        print(f"Warning: Not all projected points processed for {img_path}.")
        return None, corners_proj_cam_2D_refined
        
    return np.array(object_points_for_projector_3D_world, dtype=np.float32), corners_proj_cam_2D_refined


def calibrate_projector_as_camera(camera_images_paths, K_cam, D_cam,
                                  physical_chessboard_objp_3D, physical_board_dims,
                                  projector_pattern_points_2D, projected_pattern_dims,
                                  projector_resolution):
    """
    Calibrates the projector using images from a calibrated camera.
    """
    all_object_points_for_projector_calib = [] # 3D points in world coordinates
    all_image_points_for_projector_calib = []  # 2D points in projector's "image" coordinates

    for img_path in camera_images_paths:
        objp_for_view, _ = get_3d_points_from_camera_view_of_projected_pattern(
            img_path, K_cam, D_cam,
            physical_chessboard_objp_3D, physical_board_dims,
            projected_pattern_dims
        )
        
        if objp_for_view is not None and len(objp_for_view) == (projected_pattern_dims[0] * projected_pattern_dims[1]):
            all_object_points_for_projector_calib.append(objp_for_view)
            all_image_points_for_projector_calib.append(projector_pattern_points_2D)
        else:
            print(f"Skipping view from {img_path} due to issues.")

    if not all_object_points_for_projector_calib:
        print("No valid views processed. Projector calibration cannot proceed.")
        return False, None, None, None, None

    print(f"Calibrating projector with {len(all_object_points_for_projector_calib)} valid views.")
    ret, K_proj, D_proj, rvecs_proj, tvecs_proj = cv.calibrateCamera(
        all_object_points_for_projector_calib,
        all_image_points_for_projector_calib,
        projector_resolution,
        None, None 
    )
    return ret, K_proj, D_proj, rvecs_proj, tvecs_proj

def save_projector_params(rvecs, tvecs, K, dist, filepath):
    with open(filepath, 'w') as f:
        f.write(f"Projector Camera Matrix (K_proj):\n{K}\n")
        f.write(f"Projector Distortion Coefficients (D_proj):\n{dist}\n")
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R, _ = cv.Rodrigues(rvec)
            f.write(f"\nPose {i + 1} (Projector's pose in world coordinates):\n")
            f.write(f"Rotation Matrix (R_world_to_proj):\n{R}\n")
            f.write(f"Translation Vector (t_world_to_proj):\n{tvec}\n")

if __name__ == "__main__":
    # These must be obtained from your prior camera calibration (e.g., from a file or step_1.py)
    K_cam_calibrated = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32) # Placeholder
    D_cam_calibrated = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) # Placeholder: [k1, k2, p1, p2, k3]
    camera_images_paths_for_proj_calib = glob.glob('../Data/GrayCodes/chess/*.jpg') # Placeholder: Update this path
    
    if not K_cam_calibrated.any() or not D_cam_calibrated.any():
        print("Error: K_cam_calibrated or D_cam_calibrated is not set. Please provide camera calibration parameters.")
        exit()
    if not camera_images_paths_for_proj_calib:
        print(f"Error: No images found at the specified path for 'camera_images_paths_for_proj_calib'. Searched: '../Data/GrayCodes/chess/*.jpg'")
        exit()
    print(f"Found {len(camera_images_paths_for_proj_calib)} images for projector calibration.")

    physical_board_dims = (7, 9)
    square_size_physical_board = 0.025
    physical_chessboard_objp_3D = np.zeros((physical_board_dims[0] * physical_board_dims[1], 3), np.float32)
    physical_chessboard_objp_3D[:, :2] = np.mgrid[0:physical_board_dims[0], 0:physical_board_dims[1]].T.reshape(-1, 2) * square_size_physical_board

    projected_pattern_dims = (7, 9) 
    projector_pattern_points_2D = np.zeros((projected_pattern_dims[0] * projected_pattern_dims[1], 2), np.float32)
    projector_pattern_points_2D[:, :2] = np.mgrid[0:projected_pattern_dims[0], 0:projected_pattern_dims[1]].T.reshape(-1, 2) 

    projector_resolution = (1920, 1080) 
    output_params_filepath = '../Result/projector_parameters.txt'

    print("Starting projector calibration...")
    ret_proj, K_proj, D_proj, rvecs_proj, tvecs_proj = calibrate_projector_as_camera(
        camera_images_paths_for_proj_calib, K_cam_calibrated, D_cam_calibrated,
        physical_chessboard_objp_3D, physical_board_dims,
        projector_pattern_points_2D, projected_pattern_dims,
        projector_resolution
    )

    if ret_proj:
        print("Projector calibration successful.")
        print(f"Projector Intrinsic Matrix (K_proj):\n{K_proj}")
        print(f"Projector Distortion Coefficients (D_proj):\n{D_proj}")
        save_projector_params(rvecs_proj, tvecs_proj, K_proj, D_proj, output_params_filepath)
        print(f"Projector parameters saved to {output_params_filepath}")
    else:
        print("Projector calibration failed.")

