import numpy as np
import cv2
import glob
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

from step_2 import decode_gray_pattern, find_correspondences
from step_3 import compute_essential_matrix

def recover_pose(E, pts1, pts2, K):
    """
    Recover relative camera pose from essential matrix
    
    Parameters:
    -----------
    E: Essential matrix
    pts1, pts2: Matched points
    K: Camera intrinsic matrix
    
    Returns:
    --------
    R: Rotation matrix
    t: Translation vector
    points_3d: Triangulated 3D points
    """
    # Debug info
    print(f"Essential matrix shape: {E.shape}")
    print(f"Number of matched points: {pts1.shape[0]}")
    
    # Recover pose
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    
    # Debug info
    print(f"recoverPose mask sum: {np.sum(mask)}/{len(mask)}")
    print(f"Rotation matrix:\n{R}")
    print(f"Translation vector:\n{t}")
    
    # Check if rotation is valid (determinant should be close to 1)
    det_R = np.linalg.det(R)
    print(f"Determinant of R: {det_R}")
    if abs(det_R - 1.0) > 0.1:
        print("WARNING: Rotation matrix may be invalid!")
    
    # Normalize translation to unit length if it's too small
    t_norm = np.linalg.norm(t)
    print(f"Translation magnitude: {t_norm}")
    if t_norm < 1e-6:
        print("WARNING: Translation is nearly zero. Normalizing to unit length.")
        t = t / (t_norm + 1e-10)
    
    # Create projection matrices
    P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    
    # Convert to camera matrices
    M1 = K @ P1
    M2 = K @ P2
    
    # Debug info
    print(f"P1 shape: {P1.shape}, P2 shape: {P2.shape}")
    print(f"M1 shape: {M1.shape}, M2 shape: {M2.shape}")
    
    # Triangulate points
    points_4d = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)
    
    # Debug info
    print(f"Triangulated points shape: {points_4d.shape}")
    
    # Convert to 3D points (from homogeneous coordinates)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    points_3d = points_3d.T
    
    # Check if points are in front of both cameras
    num_points = points_3d.shape[0]
    points_in_front_cam1 = 0
    points_in_front_cam2 = 0
    
    for i in range(num_points):
        # Check if point is in front of camera 1
        if points_3d[i, 2] > 0:
            points_in_front_cam1 += 1
            
        # Transform point to camera 2 coordinates and check if it's in front
        p_cam2 = R @ points_3d[i] + t.flatten()
        if p_cam2[2] > 0:
            points_in_front_cam2 += 1
    
    print(f"Points in front of camera 1: {points_in_front_cam1}/{num_points}")
    print(f"Points in front of camera 2: {points_in_front_cam2}/{num_points}")
    
    # If most points are behind one of the cameras, we need to change the sign of the translation
    if points_in_front_cam1 < num_points/2 or points_in_front_cam2 < num_points/2:
        print("WARNING: Most points are behind one of the cameras. Flipping the pose.")
        # Try different configurations of R and t
        best_R, best_t = R, t
        best_count = points_in_front_cam1 + points_in_front_cam2
        
        # Option 1: Negate t
        neg_t = -t
        count_front1, count_front2 = 0, 0
        for i in range(num_points):
            if points_3d[i, 2] > 0:
                count_front1 += 1
            p_cam2 = R @ points_3d[i] + neg_t.flatten()
            if p_cam2[2] > 0:
                count_front2 += 1
        
        if count_front1 + count_front2 > best_count:
            best_R, best_t = R, neg_t
            best_count = count_front1 + count_front2
            print(f"Using negated t: {count_front1} + {count_front2} = {best_count} points in front")
        
        # Option 2: Rotate R by 180 degrees
        rot180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        new_R = R @ rot180
        new_t = -t
        
        # Triangulate with new pose
        P2_new = np.hstack((new_R, new_t))
        M2_new = K @ P2_new
        points_4d_new = cv2.triangulatePoints(M1, M2_new, pts1.T, pts2.T)
        points_3d_new = points_4d_new[:3, :] / points_4d_new[3, :]
        points_3d_new = points_3d_new.T
        
        count_front1, count_front2 = 0, 0
        for i in range(num_points):
            if points_3d_new[i, 2] > 0:
                count_front1 += 1
            p_cam2 = new_R @ points_3d_new[i] + new_t.flatten()
            if p_cam2[2] > 0:
                count_front2 += 1
                
        if count_front1 + count_front2 > best_count:
            best_R, best_t = new_R, new_t
            points_3d = points_3d_new
            best_count = count_front1 + count_front2
            print(f"Using rotated R and negated t: {count_front1} + {count_front2} = {best_count} points in front")
        
        # Use the best configuration
        R, t = best_R, best_t
    
    # Analyze depth range
    z_values = points_3d[:, 2]
    valid_z = z_values[(z_values > 0) & (z_values < 10000)]
    
    if len(valid_z) > 0:
        print(f"Depth statistics:")
        print(f"  Min depth: {np.min(valid_z)}")
        print(f"  Max depth: {np.max(valid_z)}")
        print(f"  Mean depth: {np.mean(valid_z)}")
        print(f"  5th percentile: {np.percentile(valid_z, 5)}")
        print(f"  95th percentile: {np.percentile(valid_z, 95)}")
    
    return R, t, points_3d

def plane_sweep_simplified(img1, img2, K, R1, t1, R2, t2, min_depth, max_depth, num_planes):
    """
    Simplified plane sweep algorithm for novel view synthesis
    
    Parameters:
    -----------
    img1, img2: Input color images
    K: Camera intrinsic matrix
    R1, t1: Rotation and translation for first camera
    R2, t2: Rotation and translation for second camera
    min_depth, max_depth: Minimum and maximum depth for plane sweep
    num_planes: Number of depth planes to use
    
    Returns:
    --------
    novel_view: Synthesized image at middle viewpoint
    depth_map: Estimated depth map
    """
    height, width = img1.shape[:2]
    
    # Create depth planes (logarithmically spaced for better precision)
    depths = np.geomspace(min_depth, max_depth, num_planes)
    
    # Create target (virtual) camera pose halfway between the two input views
    # Use proper rotation interpolation - SLERP would be better but this is a simple approximation
    target_R = np.eye(3)  # Simple interpolation - middle view is identity
    target_t = (t1 + t2) / 2  # Interpolate translation
    
    # Debug print camera positions
    print(f"Camera 1 position: {t1}")
    print(f"Camera 2 position: {t2}")
    print(f"Target camera position: {target_t}")
    
    # Prepare output arrays
    cost_volume = np.zeros((height, width, num_planes))
    novel_view = np.zeros((height, width, 3), dtype=np.uint8)
    depth_map = np.zeros((height, width))
    
    # Add validity mask to track which pixels have valid data
    valid_pixels = np.zeros((height, width), dtype=bool)
    
    # Camera projection matrices
    P1 = K @ np.hstack((R1, t1.reshape(-1, 1)))
    P2 = K @ np.hstack((R2, t2.reshape(-1, 1)))
    P_target = K @ np.hstack((target_R, target_t.reshape(-1, 1)))
    
    # Debug prints
    print(f"P1 shape: {P1.shape}, P2 shape: {P2.shape}")
    print(f"P1:\n{P1}\nP2:\n{P2}")
    
    # Create pixel coordinate grids once
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    points_2d = np.column_stack((x_grid.flatten(), y_grid.flatten(), np.ones(height * width)))
    
    # Convert to rays through pixels
    rays = np.dot(np.linalg.inv(K), points_2d.T)
    
    # For each depth plane
    for i, depth in enumerate(depths):
        print(f"Processing depth plane {i+1}/{num_planes} at depth {depth:.2f}")
        
        # Scale rays by depth to get 3D points
        points_3d = rays * depth
        
        # Convert to world coordinates (from target view to world)
        points_world = np.dot(target_R.T, points_3d - target_t.reshape(3, 1))
        
        # Project to view 1
        proj_1 = np.dot(P1, np.vstack((points_world, np.ones(points_world.shape[1]))))
        valid_proj_1 = proj_1[2] > 0  # Check if points are in front of camera
        proj_1[:, valid_proj_1] = proj_1[:, valid_proj_1] / proj_1[2:3, valid_proj_1]  # Normalize
        
        # Project to view 2
        proj_2 = np.dot(P2, np.vstack((points_world, np.ones(points_world.shape[1]))))
        valid_proj_2 = proj_2[2] > 0  # Check if points are in front of camera
        proj_2[:, valid_proj_2] = proj_2[:, valid_proj_2] / proj_2[2:3, valid_proj_2]  # Normalize
        
        # Get projected 2D coordinates
        x1, y1 = proj_1[0], proj_1[1]
        x2, y2 = proj_2[0], proj_2[1]
        
        # Check which points are within image bounds (with a margin for interpolation)
        margin = 1
        valid_1 = valid_proj_1 & (x1 >= margin) & (x1 < width-margin) & (y1 >= margin) & (y1 < height-margin)
        valid_2 = valid_proj_2 & (x2 >= margin) & (x2 < width-margin) & (y2 >= margin) & (y2 < height-margin)
        valid = valid_1 & valid_2
        
        print(f"Depth {depth:.2f}: {np.sum(valid)} valid points out of {valid.size}")
        if np.sum(valid) == 0:
            print("Warning: No valid projections at this depth!")
            continue
            
        # Sample colors from source images
        colors_1 = np.zeros((height * width, 3))
        colors_2 = np.zeros((height * width, 3))
        
        # Only process valid points
        valid_indices = np.where(valid)[0]
        
        for c in range(3):
            # Extract coordinates of valid points
            x1_valid = x1[valid_indices]
            y1_valid = y1[valid_indices]
            x2_valid = x2[valid_indices]
            y2_valid = y2[valid_indices]
            
            # Use map_coordinates for efficient interpolation
            img1_channel = img1[:,:,c]
            img2_channel = img2[:,:,c]
            
            # Stack coordinates for map_coordinates (y, x order for scipy)
            coords1 = np.vstack((y1_valid, x1_valid))
            coords2 = np.vstack((y2_valid, x2_valid))
            
            # Sample images
            sampled1 = map_coordinates(img1_channel, coords1, order=1, mode='constant', cval=0)
            sampled2 = map_coordinates(img2_channel, coords2, order=1, mode='constant', cval=0)
            
            # Store sampled colors
            colors_1[valid_indices, c] = sampled1
            colors_2[valid_indices, c] = sampled2
        
        # Compute photo-consistency (we use negative SSD - higher is better)
        diff = np.abs(colors_1 - colors_2)
        cost = -np.sum(diff * diff, axis=1)
        
        # Reshape cost to image dimensions
        cost_reshaped = np.full((height, width), -np.inf)
        cost_reshaped.flat[valid_indices] = cost[valid_indices]
        
        # Update cost volume
        cost_volume[:, :, i] = cost_reshaped
        
        # Update validity mask - a pixel is valid if it has at least one valid depth
        valid_mask = valid.reshape(height, width)
        valid_pixels = valid_pixels | valid_mask

    # Check if we have any valid pixels
    if not np.any(valid_pixels):
        print("ERROR: No valid pixels found in the entire cost volume!")
        return np.zeros_like(img1), np.zeros((height, width))
    
    # Find the best depth plane for each pixel
    best_plane_idx = np.argmax(cost_volume, axis=2)
    
    # Generate depth map and update validity
    for h in range(height):
        for w in range(width):
            if valid_pixels[h, w]:
                depth_map[h, w] = depths[best_plane_idx[h, w]]
    
    # Count valid pixels
    num_valid = np.sum(valid_pixels)
    print(f"Valid pixels in final depth map: {num_valid} ({num_valid/(height*width)*100:.2f}%)")
    
    # Generate novel view - use vectorized operations where possible
    count = 0
    for h in range(0, height, 1):
        for w in range(0, width, 1):
            if not valid_pixels[h, w]:
                continue
                
            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} pixels...")
                
            # Get estimated depth
            depth = depth_map[h, w]
            if depth <= 0:
                continue
            
            # Back-project to 3D
            p = np.array([w, h, 1])
            ray = np.dot(np.linalg.inv(K), p)
            point_3d = ray * depth
            
            # Convert to world coordinates
            point_world = np.dot(target_R.T, point_3d - target_t)
            
            # Project to source views
            p1 = np.dot(P1, np.append(point_world, 1))
            if p1[2] <= 0:  # Point is behind the camera
                continue
            p1 = p1 / p1[2]
            
            p2 = np.dot(P2, np.append(point_world, 1))
            if p2[2] <= 0:  # Point is behind the camera
                continue
            p2 = p2 / p2[2]
            
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            
            # Check if projections are within bounds with margin
            margin = 1
            if (margin <= x1 < width-margin and margin <= y1 < height-margin and 
                margin <= x2 < width-margin and margin <= y2 < height-margin):
                
                # Sample colors from source images using map_coordinates for better interpolation
                coords1 = np.array([[y1], [x1]])
                coords2 = np.array([[y2], [x2]])
                
                color1 = np.zeros(3)
                color2 = np.zeros(3)
                
                for c in range(3):
                    color1[c] = map_coordinates(img1[:,:,c], coords1, order=1)[0]
                    color2[c] = map_coordinates(img2[:,:,c], coords2, order=1)[0]
                
                # Simple blending based on proximity to cameras
                # Calculate actual 3D distances instead of just using translation
                d1 = np.linalg.norm(point_world - t1)
                d2 = np.linalg.norm(point_world - t2)
                total_dist = d1 + d2
                
                if total_dist > 0:
                    w1 = d2 / total_dist  # Weight inversely proportional to distance
                    w2 = d1 / total_dist
                else:
                    w1 = w2 = 0.5  # Equal weights if at same point
                
                # Set the pixel color
                novel_view[h, w] = np.clip(w1 * color1 + w2 * color2, 0, 255).astype(np.uint8)
    
    print(f"Total pixels filled in novel view: {count}")
    
    # Fill any remaining holes with basic inpainting
    if count < height * width:
        print("Filling holes in the novel view...")
        mask = (np.sum(novel_view, axis=2) == 0).astype(np.uint8)
        novel_view = cv2.inpaint(novel_view, mask, 3, cv2.INPAINT_TELEA)
    
    return novel_view, depth_map

if __name__ == "__main__":
    # Load grayscale images for decoding
    print("Loading images...")
    images_view0 = sorted(glob.glob('../Data/GrayCodes/view0/*.jpg'))
    images_view1 = sorted(glob.glob('../Data/GrayCodes/view1/*.jpg'))

    if len(images_view0) == 0 or len(images_view1) == 0:
        print("ERROR: No images found. Check your paths!")
        exit(1)
        
    print(f"Found {len(images_view0)} images for view 0 and {len(images_view1)} for view 1")

    images0 = []
    images1 = []
    for i in range(len(images_view0)):
        resized0 = cv2.resize(cv2.imread(images_view0[i], cv2.IMREAD_GRAYSCALE), (1920, 1080))
        resized1 = cv2.resize(cv2.imread(images_view1[i], cv2.IMREAD_GRAYSCALE), (1920, 1080))
        images0.append(resized0)
        images1.append(resized1)

    # Decode Gray codes
    print("Decoding Gray codes...")
    result0 = decode_gray_pattern(images0)
    result1 = decode_gray_pattern(images1)

    # Load color images for final visualization - use the first image from the dataset
    img1Color = cv2.resize(cv2.imread(images_view0[0], cv2.IMREAD_COLOR), (1920, 1080))
    img2Color = cv2.resize(cv2.imread(images_view1[0], cv2.IMREAD_COLOR), (1920, 1080))
    
    # Display the color images to verify they're loaded correctly
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1Color, cv2.COLOR_BGR2RGB))
    plt.title('View 1 Color Image')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2Color, cv2.COLOR_BGR2RGB))
    plt.title('View 2 Color Image')
    plt.tight_layout()
    plt.savefig('input_images.jpg')

    # Find correspondences (keypoints1, keypoints2, matches)
    print("Finding correspondences...")
    keypoints1, keypoints2, matches = find_correspondences(result0, result1)
    print(f"Found {len(matches)} correspondences")
    
    if len(matches) < 10:
        print("ERROR: Too few correspondences found. Gray code decoding may have failed.")
        exit(1)

    # Camera intrinsic matrix K
    K = np.array([
        [9.51663140e+03, 0.00000000e+00, 2.81762458e+03],
        [0.00000000e+00, 8.86527952e+03, 1.14812762e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    # Compute essential matrix, mask, matched points
    print("Computing essential matrix...")
    E, mask, pts1, pts2 = compute_essential_matrix(keypoints1, keypoints2, matches, K)
    
    # Recover camera pose with our improved function
    print("Recovering camera pose...")
    R, t, points_3d = recover_pose(E, pts1, pts2, K)
    
    # Set up first camera as the origin
    R1 = np.eye(3)
    t1 = np.zeros(3)
    
    # Set up second camera using the recovered pose
    R2 = R
    t2 = t.flatten()
    
    # Determine scene depth range from triangulated points
    z_values = points_3d[:, 2]
    valid_z = z_values[(z_values > 0) & (z_values < 10000)]  # Filter outliers
    
    if len(valid_z) > 0:
        min_depth = np.percentile(valid_z, 5)  # 5th percentile to avoid outliers
        max_depth = np.percentile(valid_z, 95)  # 95th percentile to avoid outliers
    else:
        print("WARNING: No valid depth points found. Using default depth range.")
        min_depth = 100  # Try larger default values
        max_depth = 1000
    
    print(f"Depth range: {min_depth:.2f} to {max_depth:.2f}")
    
    # Add safety margin to depth range
    depth_margin = 0.5  # 50% margin
    depth_range = max_depth - min_depth
    min_depth = max(0.1, min_depth - depth_range * depth_margin)
    max_depth = max_depth + depth_range * depth_margin
    
    print(f"Adjusted depth range: {min_depth:.2f} to {max_depth:.2f}")
    
    # Visualize 3D points to debug camera setup
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    
    # Filter points for visualization
    valid_points = points_3d[(points_3d[:, 2] > 0) & (points_3d[:, 2] < max_depth*2)]
    if len(valid_points) > 5000:  # Subsample for clearer visualization
        indices = np.random.choice(len(valid_points), 5000, replace=False)
        valid_points = valid_points[indices]
    
    ax.scatter3D(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], c='blue', s=1)
    
    # Plot cameras
    ax.scatter3D(0, 0, 0, c='red', s=100)  # Camera 1 at origin
    cam2_pos = -R2.T @ t2  # Camera 2 position in world coordinates
    ax.scatter3D(cam2_pos[0], cam2_pos[1], cam2_pos[2], c='green', s=100)
    
    # Connect cameras
    ax.plot3D([0, cam2_pos[0]], [0, cam2_pos[1]], [0, cam2_pos[2]], 'k-')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points and Camera Positions')
    plt.savefig('3d_reconstruction.jpg')
    
    # Downsample images for faster processing
    scale_factor = 0.25  # Use a smaller resolution for debugging
    h, w = img1Color.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    img1_small = cv2.resize(img1Color, (new_w, new_h))
    img2_small = cv2.resize(img2Color, (new_w, new_h))
    
    # Scale camera matrix
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_factor
    K_scaled[1, 1] *= scale_factor
    K_scaled[0, 2] *= scale_factor
    K_scaled[1, 2] *= scale_factor
    
    # Number of planes for plane sweep
    num_planes = 32
    
    # Run plane sweep with our improved algorithm
    print("Running plane sweep...")
    novel_view, depth_map = plane_sweep_simplified(
        img1_small, img2_small, K_scaled, R1, t1, R2, t2, 
        min_depth, max_depth, num_planes
    )
    
    # Check if novel view has any data
    if np.sum(novel_view) == 0:
        print("ERROR: Novel view is completely black!")
    else:
        print(f"Novel view summary - min: {novel_view.min()}, max: {novel_view.max()}, mean: {novel_view.mean():.2f}")
    
    # Check depth map
    if np.sum(depth_map) == 0:
        print("ERROR: Depth map is completely zero!")
    else:
        print(f"Depth map summary - min: {depth_map.min():.2f}, max: {depth_map.max():.2f}, mean: {depth_map.mean():.2f}")
        
        # Count valid depth values (non-zero)
        valid_depth = depth_map > 0
        print(f"Valid depth pixels: {np.sum(valid_depth)}/{depth_map.size} ({np.sum(valid_depth)/depth_map.size*100:.2f}%)")
    
    # Normalize depth map for visualization
    valid_depths = depth_map[depth_map > 0]
    if len(valid_depths) > 0:
        depth_viz = np.zeros_like(depth_map)
        depth_viz[depth_map > 0] = (depth_map[depth_map > 0] - valid_depths.min()) / (valid_depths.max() - valid_depths.min() + 1e-10)
        depth_viz = (depth_viz * 255).astype(np.uint8)
        depth_viz_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
    else:
        depth_viz_color = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    # Create composite visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1_small, cv2.COLOR_BGR2RGB))
    plt.title('Input View 1')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2_small, cv2.COLOR_BGR2RGB))
    plt.title('Input View 2')
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(novel_view, cv2.COLOR_BGR2RGB))
    plt.title('Novel View (Middle)')
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(depth_viz_color, cv2.COLOR_BGR2RGB))
    plt.title('Depth Map')
    
    plt.tight_layout()
    plt.savefig('results_comparison.jpg')
    
    # Save individual results
    cv2.imwrite('novel_view.jpg', novel_view)
    cv2.imwrite('depth_map.jpg', depth_viz_color)
    
    print("Novel view synthesis complete!")
    print("Results saved as novel_view.jpg and depth_map.jpg")