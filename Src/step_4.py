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
                print("R_new calculated")
                
                # Recompute 3D points with the interpolated camera pose
                new_points_3D = triangulate_opencv(pts1, pts2, K, R_new, T_new)
                print("new_points_3D calculated")
                
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
                    print("Creating depth map with valid points")
                    depth_map = create_depth_map(new_points_3D, colors, img_shape, pts1)
                
                depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                print("Depth map colored")
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

    keypoints1, keypoints2, matches = find_correspondences(result0, result1)

    K = np.array([[9.51663140e+03, 0.00000000e+00, 2.81762458e+03],[0.00000000e+00, 8.86527952e+03, 1.14812762e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    E, mask, pts1, pts2 = compute_essential_matrix(keypoints1, keypoints2, matches, K)

    # R = rotatie vector
    # T = translatie vector
    _, R, T, mask = cv2.recoverPose(E, pts1, pts2, K)

    points_3D_opencv = triangulate_opencv(pts1, pts2, K, R, T)
    points_3D_manual = triangulate_manual(pts1, pts2, K, R, T)

    colors = get_colors(img1Color, img2Color, pts1, pts2)

    depthMap = create_depth_map(points_3D_manual, colors, img1Color.shape, pts1)
    visualize_depth_map(depthMap)

    switch_camera_view_static(R, T, K, pts1, pts2, points_3D_manual, colors, img1Color.shape)
    switch_camera_view_dynamic(R, T, K, pts1, pts2, points_3D_manual, colors, img1Color.shape)

