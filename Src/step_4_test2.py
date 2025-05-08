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

# Create a depth map from the 3D points
def create_depth_map(points_3D, colors, img_shape, pts1):
    """
    Create a depth map from the 3D points.
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
    """
    plt.imshow(depthMap, cmap='gray')
    plt.colorbar()
    plt.show()

# Create color scale legend
def add_color_scale(image):
    # Create gradient bar
    height, width = image.shape[:2]
    legend_height = 30
    legend_width = width - 20  # Padding on both sides
    legend_x = 10
    legend_y = height - legend_height - 10
    
    # Create gradient
    gradient = np.linspace(0, 255, legend_width).astype(np.uint8)
    gradient = np.tile(gradient, (legend_height, 1))
    
    # Apply colormap to gradient
    gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    
    # Add text labels for minimum and maximum depths
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Near", (legend_x, legend_y - 5), 
                font, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Far", (legend_x + legend_width - 30, legend_y - 5), 
                font, 0.5, (255, 255, 255), 1)
    
    # Overlay the gradient onto the image
    image[legend_y:legend_y+legend_height, legend_x:legend_x+legend_width] = gradient_colored[0:legend_height, 0:legend_width]
    
    # Draw border around legend
    cv2.rectangle(image, (legend_x, legend_y), (legend_x+legend_width, legend_y+legend_height), (255, 255, 255), 1)
    
    return image

def interpolate_camera_view_static(R1, R2, t1, t2, alpha, lateral_offset=0):
    # Convert matrices to Rotation objects
    r1 = ScipyRotation.from_matrix(R1)
    r2 = ScipyRotation.from_matrix(R2)
    
    # Create a consolidated rotations object with both rotations
    key_rots = ScipyRotation.from_matrix(np.stack([R1, R2]))
    
    # Create times for the two rotations
    key_times = [0, 1]
    
    # Create a Slerp object
    slerp = Slerp(key_times, key_rots)
    
    # Get interpolated rotation at the given alpha
    r_interp = slerp([alpha])[0]
    
    # Interpolate translation
    t_interp = (1 - alpha) * t1 + alpha * t2
    
    # Apply lateral movement (perpendicular to viewing direction)
    if lateral_offset != 0:
        # Get the current view direction (Z-axis of the camera)
        view_dir = r_interp.as_matrix()[:, 2]
        
        # Get the right vector (X-axis of the camera)
        right_vector = r_interp.as_matrix()[:, 0]
        
        # Apply lateral offset along the right vector
        t_interp = t_interp + lateral_offset * right_vector.reshape(3, 1)
    
    return r_interp.as_matrix(), t_interp

def switch_camera_view_static(R, T, K, pts1, pts2, points_3D, colors, img_shape):
    alpha = 0.0       # Forward/backward interpolation factor
    lateral = 0.0     # Left/right movement factor
    R_base, T_base = np.eye(3), np.zeros((3, 1))  # Initial camera pose
    
    # Create initial depth map
    depth_map = create_depth_map(points_3D, colors, img_shape, pts1)
    depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    
    print("Depth map viewer started. Press A/D to move forward/backward, W/S to move left/right, ESC to exit.")
    
    change_view = False  # Flag to indicate if view needs to be updated
    
    while True:
        # Display current depth map with instructions
        info_display = depth_map_colored.copy()
        cv2.putText(info_display, "A/D: Forward/Back, W/S: Left/Right, ESC: Exit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_display, f"Alpha: {alpha:.1f}, Lateral: {lateral:.3f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add color scale legend
        info_display = add_color_scale(info_display)
        
        cv2.imshow("Depth Map", info_display)
        
        # Use a shorter wait time for more responsive keyboard input
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC to exit
            print("Exiting depth map viewer.")
            break
            
        elif key == ord('d'):  # Move forward
            print("Moving forward. Alpha =", min(alpha + 0.1, 1.0))
            alpha = min(alpha + 0.1, 1.0)
            change_view = True
            
        elif key == ord('a'):  # Move backward
            print("Moving backward. Alpha =", max(alpha - 0.1, 0.0))
            alpha = max(alpha - 0.1, 0.0)
            change_view = True
            
        elif key == ord('w'):  # Move left
            print("Moving left. Lateral =", lateral - 0.1)
            lateral -= 0.1
            change_view = True
            
        elif key == ord('s'):  # Move right
            print("Moving right. Lateral =", lateral + 0.1)
            lateral += 0.1
            change_view = True
        
        # Only update view when needed (when a key is pressed)
        if change_view:
            print("key pressed")
            try:
                # Update the view with both forward/backward and lateral movement
                R_new, T_new = interpolate_camera_view_static(R_base, R, T_base, T.reshape(3, 1), alpha, lateral)
                
                # Recompute 3D points with the interpolated camera pose
                new_points_3D = triangulate_opencv(pts1, pts2, K, R_new, T_new)
                
                # Check if triangulation produced valid points
                if np.isnan(new_points_3D).any() or np.isinf(new_points_3D).any():
                    print("Warning: Some points have invalid coordinates (NaN or Inf)")
                    # Filter out invalid points
                    valid_indices = ~(np.isnan(new_points_3D).any(axis=1) | np.isinf(new_points_3D).any(axis=1))
                    new_points_3D = new_points_3D[valid_indices]
                    colors_filtered = colors[valid_indices]
                    pts1_filtered = pts1[valid_indices]
                    depth_map = create_depth_map(new_points_3D, colors_filtered, img_shape, pts1_filtered)
                else:
                    depth_map = create_depth_map(new_points_3D, colors, img_shape, pts1)
                
                depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                change_view = False  # Reset flag after update
                
            except Exception as e:
                print(f"Error during camera view update: {e}")
                # If there's an error, reset to original position
                alpha = 0.0
                lateral = 0.0
                depth_map = create_depth_map(points_3D, colors, img_shape, pts1)
                depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                change_view = False
    
    # Clean up properly to ensure window closes
    cv2.destroyWindow("Depth Map")
    cv2.waitKey(1)  # Give time for window destruction
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def interpolate_camera_view_dynamic(R1, R2, t1, t2, alpha, lateral_offset=0):
    # Convert matrices to Rotation objects
    r1 = ScipyRotation.from_matrix(R1)
    r2 = ScipyRotation.from_matrix(R2)
    
    # Create a consolidated rotations object with both rotations
    key_rots = ScipyRotation.from_matrix(np.stack([R1, R2]))
    
    # Create times for the two rotations
    key_times = [0, 1]
    
    # Create a Slerp object
    slerp = Slerp(key_times, key_rots)
    
    # Get interpolated rotation at the given alpha
    r_interp = slerp([alpha])[0]
    
    # Interpolate translation
    t_interp = (1 - alpha) * t1 + alpha * t2
    
    # Apply lateral movement (perpendicular to viewing direction)
    if lateral_offset != 0:
        # Get the current view direction (Z-axis of the camera)
        view_dir = r_interp.as_matrix()[:, 2]
        
        # Get the right vector (X-axis of the camera)
        right_vector = r_interp.as_matrix()[:, 0]
        
        # Apply lateral offset along the right vector
        t_interp = t_interp + lateral_offset * right_vector.reshape(3, 1)
    
    return r_interp.as_matrix(), t_interp

def switch_camera_view_dynamic(R, T, K, pts1, pts2, points_3D, colors, img_shape):
    alpha = 0.0       # Forward/backward interpolation factor
    lateral = 0.0     # Left/right movement factor
    R_base, T_base = np.eye(3), np.zeros((3, 1))  # Initial camera pose
    
    # Create initial depth map
    depth_map = create_depth_map(points_3D, colors, img_shape, pts1)
    depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    
    print("Depth map viewer started. Press A/D to move forward/backward, W/S to move left/right, ESC to exit.")
    
    change_view = False  # Flag to indicate if view needs to be updated
    
    # Store original parameters
    original_points_3D = points_3D.copy()
    
    while True:
        # Display current depth map with instructions
        info_display = depth_map_colored.copy()
        cv2.putText(info_display, "A/D: Forward/Back, W/S: Left/Right, ESC: Exit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_display, f"Alpha: {alpha:.1f}, Lateral: {lateral:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add color scale legend
        info_display = add_color_scale(info_display)
        
        cv2.imshow("Depth Map", info_display)
        
        # Use a shorter wait time for more responsive keyboard input
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC to exit
            print("Exiting depth map viewer.")
            break
            
        elif key == ord('d'):  # Move forward
            print("Moving forward. Alpha =", min(alpha + 0.1, 1.0))
            alpha = min(alpha + 0.1, 1.0)
            change_view = True
            
        elif key == ord('a'):  # Move backward
            print("Moving backward. Alpha =", max(alpha - 0.1, 0.0))
            alpha = max(alpha - 0.1, 0.0)
            change_view = True
            
        elif key == ord('w'):  # Move left
            print("Moving left. Lateral =", lateral - 0.25)
            lateral -= 0.025
            change_view = True
            
        elif key == ord('s'):  # Move right
            print("Moving right. Lateral =", lateral + 0.25)
            lateral += 0.025
            change_view = True
        
        # Only update view when needed (when a key is pressed)
        if change_view:
            try:
                # Update the view with both forward/backward and lateral movement
                R_new, T_new = interpolate_camera_view_dynamic(R_base, R, T_base, T.reshape(3, 1), alpha, lateral)
                
                # Instead of triangulating, transform the original 3D points to the new viewpoint
                # This is more efficient and gives a better sense of the actual camera movement
                
                # Create the transformation matrix from the new camera pose
                R_transform = R_new.T  # Transpose for inverse rotation
                T_transform = -np.dot(R_transform, T_new)  # Translation in the rotated frame
                
                # Create 4x4 transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = R_transform
                transform[:3, 3] = T_transform.flatten()
                
                # Apply transformation to original 3D points
                # Convert to homogeneous coordinates
                points_homogeneous = np.hstack((original_points_3D, np.ones((original_points_3D.shape[0], 1))))
                transformed_points = np.dot(points_homogeneous, transform.T)
                new_points_3D = transformed_points[:, :3]  # Convert back to 3D
                
                # Project the transformed 3D points to 2D to get new pixel locations
                projected_pts = []
                for point in new_points_3D:
                    # Project 3D point to 2D using the camera matrix
                    point_homogeneous = np.append(point, 1)
                    projected = np.dot(K, point_homogeneous[:3])
                    if projected[2] != 0:  # Avoid division by zero
                        x, y = projected[0] / projected[2], projected[1] / projected[2]
                        projected_pts.append([x, y])
                    else:
                        projected_pts.append([0, 0])  # Fallback for invalid points
                
                projected_pts = np.array(projected_pts)
                
                # Create depth map from transformed points and projected coordinates
                depth_map = np.zeros(img_shape[:2], dtype=np.float32)
                for (x, y), point, color in zip(projected_pts.astype(int), new_points_3D, colors):
                    if 0 <= y < img_shape[0] and 0 <= x < img_shape[1]:
                        # Use transformed Z coordinate for depth
                        depth_map[y, x] = point[2]
                
                # Normalize the depth map
                depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                change_view = False  # Reset flag after update
                
            except Exception as e:
                print(f"Error during camera view update: {e}")
                # If there's an error, reset to original position
                alpha = 0.0
                lateral = 0.0
                depth_map = create_depth_map(points_3D, colors, img_shape, pts1)
                depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                change_view = False
    
    # Clean up properly to ensure window closes
    cv2.destroyWindow("Depth Map")
    cv2.waitKey(1)  # Give time for window destruction
    cv2.destroyAllWindows()
    cv2.waitKey(1)

import numpy as np

def compute_scene_center_depth(points_3d, projection_matrix):
    """
    Computes the center of the point cloud and its depth in the camera view.

    Args:
        points_3d (np.ndarray): Nx3 array of 3D points.
        projection_matrix (np.ndarray): 3x4 or 4x4 projection matrix.

    Returns:
        float: The depth of the center point.
    """
    # Step 1: Compute the centroid of the point cloud
    centroid = np.mean(points_3d, axis=0)  # Shape: (3,)

    # Step 2: Convert to homogeneous coordinates
    centroid_h = np.append(centroid, 1)  # Shape: (4,)

    # Step 3: Project the centroid using the projection matrix
    projected = projection_matrix @ centroid_h  # Shape: (3,) or (4,)

    # Step 4: Convert from homogeneous if necessary
    if projected.shape[0] == 4:
        depth = projected[2] / projected[3]
    else:
        depth = projected[2]  # already normalized or assumed perspective division handled elsewhere

    return depth

def deproject_image_corners(intrinsics, extrinsics, width, height, depth):
    """
    Deprojects the image plane corners into 3D space at a given depth.

    Args:
        intrinsics (np.ndarray): 3x3 camera intrinsics matrix.
        extrinsics (np.ndarray): 4x4 camera-to-world matrix (R|t).
        width (int): Image width.
        height (int): Image height.
        depth (float): Depth value to deproject onto.

    Returns:
        np.ndarray: 4x3 array of 3D corner points on the depth plane (in world coordinates).
    """
    # Step 1: Define 2D image corners in pixel coordinates
    image_corners = np.array([
        [0, 0],               # top-left
        [width - 1, 0],       # top-right
        [0, height - 1],      # bottom-left
        [width - 1, height - 1]  # bottom-right
    ])  # shape: (4, 2)

    # Step 2: Convert corners to normalized camera coordinates
    inv_K = np.linalg.inv(intrinsics)
    corners_homog = np.concatenate([image_corners, np.ones((4, 1))], axis=1).T  # shape: (3, 4)
    rays = inv_K @ corners_homog  # shape: (3, 4)

    # Step 3: Scale rays to the given depth
    rays = rays * depth / rays[2, :]  # normalize z to depth

    # Step 4: Convert to world coordinates using extrinsics
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_3d = (R @ rays + t[:, np.newaxis]).T  # shape: (4, 3)

    return points_3d

def project_points_to_image(points_3d, intrinsics, extrinsics):
    """
    Projects 3D points onto the image plane using intrinsics and extrinsics.

    Args:
        points_3d (np.ndarray): 4x3 3D points in world coordinates.
        intrinsics (np.ndarray): 3x3 camera intrinsics matrix.
        extrinsics (np.ndarray): 4x4 world-to-camera matrix.

    Returns:
        np.ndarray: 4x2 projected 2D points.
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

    Args:
        image (np.ndarray): Input image.
        depth_plane_3d (np.ndarray): 4x3 array of 3D corners of the plane.
        camera_intrinsics (np.ndarray): 3x3 intrinsics of the input camera.
        camera_extrinsics (np.ndarray): 4x4 world-to-camera matrix.

    Returns:
        np.ndarray: Warped image aligned with the depth plane.
    """
    h, w = image.shape[:2]

    # Step 1: Image corners (2D)
    image_corners = np.array([
        [0, 0],         # top-left
        [w - 1, 0],     # top-right
        [0, h - 1],     # bottom-left
        [w - 1, h - 1]  # bottom-right
    ], dtype=np.float32)

    # Step 2: Project 3D plane corners to input camera image
    projected_2d = project_points_to_image(depth_plane_3d, camera_intrinsics, camera_extrinsics).astype(np.float32)

    # Step 3: Compute homography from input image to projected plane
    H, _ = cv2.findHomography(image_corners, projected_2d)

    # Step 4: Warp image using the homography
    warped_image = cv2.warpPerspective(image, H, (w, h))

    return warped_image




import cv2
import numpy as np

def compute_best_depth_image(warped_left_list, warped_right_list):
    """
    Computes the final interpolated image by minimizing pixel-wise error across depth layers.

    Args:
        warped_left_list (list of np.ndarray): List of warped left images at each depth.
        warped_right_list (list of np.ndarray): List of warped right images at each depth.

    Returns:
        np.ndarray: Final image composed of best-pixel matches.
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

        # Apply slight blur to reduce noise/artifacts
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
    num_pts = points_3d.shape[0]
    homog = np.hstack([points_3d, np.ones((num_pts, 1))])
    transformed = (transform @ homog.T).T
    return transformed[:, :3]


def make_warped_lists(K, R, T, img_left, img_right, width, height, depth_center):
    warped_left_list = []
    warped_right_list = []

    # Left camera at origin (identity transform)
    extr_left = np.eye(4)
    extr_left[:3, :3] = np.eye(3)  # No rotation
    extr_left[:3, 3] = np.zeros(3)  # At origin

    extr_right = np.eye(4)
    extr_right[:3, :3] = R
    extr_right[:3, 3] = T.squeeze()

    midpoint_translation = (extr_left[:3, 3] + extr_right[:3, 3]) / 2

    extr_virtual = np.eye(4)
    extr_virtual[:3, :3] = np.eye(3)  # Keep same orientation (optional)
    extr_virtual[:3, 3] = midpoint_translation

    num_layers = 50
    depth_range = 0.1  
    depths = np.linspace(depth_center * (1 - depth_range),
                         depth_center * (1 + depth_range),
                         num_layers)
    
    black_count_left = 0
    black_count_right = 0
    for d in depths:
        # (1) Deproject corners in virtual camera frame
        depth_plane_3d_virtual = deproject_image_corners(K, extr_virtual, width, height, d)

        # (2) Compute relative transforms
        T_virtual_to_left = extr_left @ np.linalg.inv(extr_virtual)
        T_virtual_to_right = extr_right @ np.linalg.inv(extr_virtual)

        # (3) Transform depth plane into left and right camera coordinates
        depth_plane_3d_left = apply_transform(depth_plane_3d_virtual, T_virtual_to_left)
        depth_plane_3d_right = apply_transform(depth_plane_3d_virtual, T_virtual_to_right)

        # (4) Warp images to the virtual depth plane
        warped_left = warp_image_to_depth_plane(img_left, depth_plane_3d_left, K, extr_left)
        warped_right = warp_image_to_depth_plane(img_right, depth_plane_3d_right, K, extr_right)


        # Check if the warped images are empty or have unusual pixel values
        if np.all(warped_left == 0):
            black_count_left += 1 
            print(f"Warning: warped_left is all black at depth {d}")
        if np.all(warped_right == 0):
            black_count_right += 1
            print(f"Warning: warped_right is all black at depth {d}")

        # Show the warped images for debugging
        cv2.imshow("Warped Left", warped_left)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Warped Right", warped_right)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Add to lists
        warped_left_list.append(warped_left)
        warped_right_list.append(warped_right)

    print(f"Number of black warped left images: {black_count_left}")
    print(f"Number of black warped right images: {black_count_right}")
    return warped_left_list, warped_right_list

import numpy as np
import cv2

def render_virtual_view(warped_left_list, warped_right_list):
    """
    Renders the final virtual camera image from warped left and right image stacks.

    Args:
        warped_left_list (list of np.ndarray): List of left images warped to virtual camera.
        warped_right_list (list of np.ndarray): List of right images warped to virtual camera.

    Returns:
        np.ndarray: Final rendered image from virtual viewpoint.
    """
    num_layers = len(warped_left_list)
    h, w, c = warped_left_list[0].shape

    # Initialize output
    min_error = np.full((h, w), np.inf)
    best_image = np.zeros((h, w, c), dtype=np.uint8)

    for i in range(num_layers):
        left = warped_left_list[i]
        right = warped_right_list[i]

        # Compute pixel-wise error (grayscale absdiff)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        error = cv2.absdiff(gray_left, gray_right).astype(np.float32)

        # Mask where this layer is better
        better_mask = error < min_error
        min_error[better_mask] = error[better_mask]

        # Average color (or just pick left/right – here we average)
        blended = ((left.astype(np.uint16) + right.astype(np.uint16)) // 2).astype(np.uint8)

        # Update best image where this depth is better
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

    # depthMap = create_depth_map(points_3D_opencv, colors, img1Color.shape, pts1)
    # visualize_depth_map(depthMap)

    # switch_camera_view_static(R, T, K, pts1, pts2, points_3D_opencv, colors, img1Color.shape)
    # switch_camera_view_dynamic(R, T, K, pts1, pts2, points_3D_opencv, colors, img1Color.shape)

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
    cv2.imwrite("../Result/final_output_plane_sweep1.png", final_output)
    cv2.destroyAllWindows()



