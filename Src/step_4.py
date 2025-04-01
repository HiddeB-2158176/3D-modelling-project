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

# def plane_sweep(R, T, K, img1, img2, disparity_range=20):
#     """
#     Perform plane sweeping to create a depth map.
    
#     Parameters:
#     - R: Rotation matrix of the second camera.
#     - T: Translation vector of the second camera.
#     - K: Camera intrinsic matrix.
#     - img1: Left image (grayscale).
#     - img2: Right image (grayscale).
#     - disparity_range: The number of disparity levels to sweep over.
    
#     Returns:
#     - depth_map: Computed depth map using plane sweeping.
#     """

#     # Image dimensions
#     h, w = img1.shape[:2]

#     # Initialize depth and cost volume
#     depth_map = np.zeros((h, w), dtype=np.float32)
#     cost_volume = np.full((h, w, disparity_range), np.inf)  # Store cost for each depth hypothesis

#     # Define depth values (inverse depth values for proper disparity)
#     depth_values = np.linspace(0.1, 2.0, disparity_range)  # Adjust based on scene scale

#     # Convert the extrinsic parameters into a projection matrix
#     P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera (identity pose)
#     P2 = K @ np.hstack((R, T))  # Second camera (transformed pose)

#     for d_idx, depth in enumerate(depth_values):
#         # Create a disparity map for the current depth
#         disparity = 1.0 / depth  # Simulating inverse depth

#         # Compute the transformation for the corresponding disparity
#         pts1_homogeneous = np.vstack((np.indices((h, w)).reshape(2, -1), np.ones((1, h * w))))
#         pts1_3D = np.linalg.inv(K) @ pts1_homogeneous * depth  # Convert pixels to 3D

#         # Project points onto second image
#         pts2_3D = R @ pts1_3D + T
#         pts2_homogeneous = K @ pts2_3D
#         pts2 = pts2_homogeneous[:2] / pts2_homogeneous[2]  # Normalize homogeneous coordinates

#         # Reshape points back into image format
#         pts1_reshaped = pts1_homogeneous[:2].reshape(2, h, w).astype(np.float32)
#         pts2_reshaped = pts2.reshape(2, h, w).astype(np.float32)

#         # Compute image similarity using SSD (Sum of Squared Differences)
#         warped_img2 = cv2.remap(img2, pts2_reshaped[0], pts2_reshaped[1], cv2.INTER_LINEAR)
#         ssd = (img1 - warped_img2) ** 2  # SSD cost metric

#         # Store the cost at the current depth level
#         cost_volume[:, :, d_idx] = ssd

#     # Find the best depth by selecting the lowest cost per pixel
#     best_depth_idx = np.argmin(cost_volume, axis=2)
#     depth_map = depth_values[best_depth_idx]  # Convert index to actual depth values

#     # Normalize for visualization
#     depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     return depth_map

import numpy as np
import cv2

def compute_scene_depth_bounds(point_cloud):
    """
    Computes the depth range based on the center of the point cloud.
    This helps in defining the depth planes.
    
    Args:
        point_cloud (numpy array): Nx3 array representing the point cloud.

    Returns:
        depth_min (float): Minimum depth.
        depth_max (float): Maximum depth.
    """
    center = np.mean(point_cloud, axis=0)  # Compute center of the scene
    depths = point_cloud[:, 2]  # Extract Z (depth) values

    depth_min = np.min(depths)
    depth_max = np.max(depths)

    return depth_min, depth_max, center[2]  # Return depth range and center depth

def generate_depth_planes(depth_min, depth_max, center_depth, num_planes=50, offset_factor=0.5):
    """
    Generate depth planes based on center of the point cloud.

    Args:
        depth_min (float): Minimum depth value.
        depth_max (float): Maximum depth value.
        center_depth (float): Depth value at the center of the scene.
        num_planes (int): Number of depth planes to sample.
        offset_factor (float): Factor to control how far the planes extend from the center.

    Returns:
        depths (numpy array): List of depth values for planes.
    """
    # Calculate an offset around the center depth
    depth_offset = (depth_max - depth_min) * offset_factor
    
    # Generate depth planes around the center
    depths = np.linspace(center_depth - depth_offset, center_depth + depth_offset, num_planes)
    
    return depths


def back_project_to_3D(image_corners, camera_matrix, depth):
    """
    Back-projects 2D image points to a 3D plane at a given depth.

    Args:
        image_corners (numpy array): 2D pixel coordinates of the image corners.
        camera_matrix (numpy array): Intrinsic camera matrix.
        depth (float): Depth plane at which to back-project.

    Returns:
        numpy array: 3D coordinates of the projected points.
    """
    K_inv = np.linalg.inv(camera_matrix)  # Invert camera matrix
    homog_corners = np.hstack((image_corners, np.ones((4, 1))))  # Convert to homogeneous
    world_points = (K_inv @ homog_corners.T).T  # Convert to normalized camera space
    world_points *= depth  # Scale by depth

    return world_points

def compute_homography(K, depth):
    """
    Computes the homography matrix for a given camera intrinsic matrix K and depth plane.
    """

    # Define 4 canonical 2D points (image plane corners in homogeneous coordinates)
    h, w = 480, 640  # Example image dimensions, replace with actual image size
    src_points = np.array([
        [0, 0], [w, 0], [w, h], [0, h]  # Image corners
    ], dtype=np.float32)

    # Convert these 2D points to homogeneous coordinates
    src_points_h = np.hstack([src_points, np.ones((4, 1))])

    # Convert image plane points into 3D by back-projecting with depth
    src_points_3D = np.linalg.inv(K) @ src_points_h.T  # Back-project to 3D
    src_points_3D *= depth  # Scale by the depth to place on depth plane

    # Project these 3D points onto the new virtual camera plane
    dst_points = K @ src_points_3D  # Reproject into 2D
    dst_points = (dst_points[:2] / dst_points[2]).T  # Convert from homogeneous to 2D

    # Ensure we have at least 4 valid points
    if len(src_points) < 4 or len(dst_points) < 4:
        raise ValueError("Not enough points for homography computation!")

    print("Source points:\n", src_points)
    print("Destination points:\n", dst_points)
    # Compute homography
    H, status = cv2.findHomography(src_points, dst_points)
    print(f"Depth: {depth}, Homography H: {H}")

    return H

def warp_image(image, H, output_size):
    """
    Warp an image using the given homography matrix.

    Args:
        image (numpy array): Input image.
        H (numpy array): Homography matrix.
        output_size (tuple): Output image dimensions (width, height).

    Returns:
        numpy array: Warped image.
    """
    return cv2.warpPerspective(image, H, output_size)

def compute_reprojection_error(image1, image2):
    """
    Compute pixel-wise absolute difference between two images.

    Args:
        image1 (numpy array): First image.
        image2 (numpy array): Second image.

    Returns:
        numpy array: Absolute difference image.
    """
    return cv2.absdiff(image1, image2)

def sweep_depth_planes(img1, img2, K, point_cloud, num_planes=10):
    h, w, _ = img1.shape  # Ensure input images are color (HxWx3)

    # Compute the depth bounds and center
    depth_min, depth_max, center_depth = compute_scene_depth_bounds(point_cloud)
    
    # Generate depth planes based on the center of the point cloud
    depth_planes = generate_depth_planes(depth_min, depth_max, center_depth, num_planes)
    
    best_errors = np.full((h, w), np.inf)  # Track minimum error per pixel
    best_colors = np.zeros((h, w, 3), dtype=np.uint8)  # Store best color reconstruction

    for depth in depth_planes:
        # Step 1: Compute homography and warp img2 to img1 viewpoint at this depth
        H = compute_homography(K, depth)  # Implement based on plane-sweeping math
        warped_img2 = cv2.warpPerspective(img2, H, (w, h))  # Align to reference image
        warped_img2_resized = cv2.resize(warped_img2, (960, 540)) 
        cv2.imshow("Warped Image", warped_img2_resized)  # Display warped image for debugging
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Step 2: Compute pixel-wise difference (error)
        error = cv2.absdiff(img1, warped_img2)  # Difference between left and warped right
        total_error = np.sum(error, axis=2)  # Sum over RGB channels
        
        # Step 3: Update best colors where error is minimal
        mask = total_error < best_errors  # Pixels where this depth gives lower error
        best_errors[mask] = total_error[mask]  # Update minimal error map
        best_colors[mask] = warped_img2[mask]  # Assign the best-matching color pixels

    return best_colors  # Final reconstructed virtual viewpoint


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

    #depthMap = create_depth_map(points_3D_opencv, colors, img1Color.shape, pts1)
    #visualize_depth_map(depthMap)

    # switch_camera_view_static(R, T, K, pts1, pts2, points_3D_manual, colors, img1Color.shape)
    # switch_camera_view_dynamic(R, T, K, pts1, pts2, points_3D_manual, colors, img1Color.shape)

    depth_map = sweep_depth_planes(img1ColorRGB, img2ColorRGB, K, points_3D_opencv, num_planes=10)
    depth_map_colored = cv2.applyColorMap(depth_map.astype(np.uint8), cv2.COLORMAP_JET)
    depth_map_resized = cv2.resize(depth_map, (960, 540))
    plt.imshow(depth_map_resized)
    #plt.colorbar()
    plt.show()
    #visualize_depth_map(depth_map)