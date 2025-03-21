import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import GrayCodeEncoder
import random
import open3d as o3d
import os

def validate_pixels(white, black, threshold=30):
    """
    Validate pixels based on intensity difference between white and black reference images.
    Only pixels with sufficient contrast are valid for Gray code decoding.
    """
    # Convert to grayscale if images are color
    if len(white.shape) == 3:
        white_gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
        black_gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
    else:
        white_gray = white
        black_gray = black
    
    # Calculate difference
    diff = cv2.absdiff(white_gray, black_gray)
    
    # Create binary mask where difference is above threshold
    mask = np.zeros_like(diff, dtype=np.uint8)
    mask[diff > threshold] = 1
    
    return mask

def decode_bit(normal, inverse, mask, threshold=30):
    """
    Decode a single bit from normal and inverse pattern pair.
    """
    # Convert to grayscale if images are color
    if len(normal.shape) == 3:
        normal_gray = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)
        inverse_gray = cv2.cvtColor(inverse, cv2.COLOR_BGR2GRAY)
    else:
        normal_gray = normal
        inverse_gray = inverse
    
    # Calculate absolute difference
    diff = cv2.absdiff(normal_gray, inverse_gray)
    
    # Initialize bit values (default is uncertain/invalid)
    bit = np.ones_like(normal_gray, dtype=np.int8) * -1
    
    # Set bit to 1 where normal > inverse (with threshold)
    bit[(normal_gray > inverse_gray + threshold) & (mask == 1)] = 1
    
    # Set bit to 0 where inverse > normal (with threshold)
    bit[(inverse_gray > normal_gray + threshold) & (mask == 1)] = 0
    
    # Return the bit values
    return bit

def decode_gray_patterns(h_patterns, v_patterns, white, black, threshold=30):
    """
    Decode Gray code patterns to get unique identifiers for each pixel.
    Preserves color information from the white image.
    """
    height, width = black.shape[:2]
    
    # Validate pixels using white and black reference images
    valid_mask = validate_pixels(white, black, threshold)
    
    # Initialize arrays for horizontal and vertical codes
    h_code = np.zeros((height, width), dtype=np.uint32)
    v_code = np.zeros((height, width), dtype=np.uint32)
    
    # Process horizontal patterns
    for i in range(0, len(h_patterns), 2):
        if i+1 < len(h_patterns):
            normal = h_patterns[i]
            inverse = h_patterns[i+1]
            bit_index = i // 2
            
            # Decode bit
            bit = decode_bit(normal, inverse, valid_mask, threshold)
            
            # Add to horizontal code
            h_code[bit == 1] |= (1 << bit_index)
            
            # Mark invalid bits
            valid_mask[bit == -1] = 0
    
    # Process vertical patterns
    for i in range(0, len(v_patterns), 2):
        if i+1 < len(v_patterns):
            normal = v_patterns[i]
            inverse = v_patterns[i+1]
            bit_index = i // 2
            
            # Decode bit
            bit = decode_bit(normal, inverse, valid_mask, threshold)
            
            # Add to vertical code
            v_code[bit == 1] |= (1 << bit_index)
            
            # Mark invalid bits
            valid_mask[bit == -1] = 0
    
    # Convert Gray code to binary
    h_code = gray_to_binary(h_code)
    v_code = gray_to_binary(v_code)
    
    # Create unique identifiers by combining horizontal and vertical codes
    identifier = (h_code << 16) | v_code
    
    # Store 3D points with color
    identifier_list = []
    for y in range(height):
        for x in range(width):
            if valid_mask[y, x] == 1:
                # Extract color from white image (which should have full illumination)
                if len(white.shape) == 3:
                    color = white[y, x].copy()  # BGR format
                else:
                    color = np.array([255, 255, 255])  # Default white if grayscale
                
                identifier_list.append((x, y, identifier[y, x], color))
    
    return identifier_list, h_code, v_code, valid_mask

def gray_to_binary(gray):
    """
    Convert Gray code to binary.
    """
    binary = gray.copy()
    mask = binary
    while mask.any():
        mask = mask >> 1
        binary ^= mask
    return binary

def find_correspondences(identifier_list1, identifier_list2):
    """
    Find pixel correspondences between two views based on identifiers.
    """
    # Create dictionary for fast lookup in second view
    id_dict = {identifier: (x, y, color) for x, y, identifier, color in identifier_list2}
    
    matches = []
    keypoints1, keypoints2, colors = [], [], []
    
    # Find matches
    for x1, y1, identifier, color1 in identifier_list1:
        if identifier in id_dict and identifier > 0:
            x2, y2, color2 = id_dict[identifier]
            
            # Add keypoints and match
            keypoints1.append(cv2.KeyPoint(x1, y1, 1))
            keypoints2.append(cv2.KeyPoint(x2, y2, 1))
            matches.append(cv2.DMatch(len(keypoints1)-1, len(keypoints2)-1, 0))
            colors.append(color1)  # Use color from first view
    
    print(f"Found {len(matches)} correspondences")
    return keypoints1, keypoints2, matches, colors

def draw_matched_correspondences(img1, img2, keypoints1, keypoints2, matches, max_matches=100):
    """
    Draw a subset of correspondences for visualization.
    """
    # Sample a subset of matches for clearer visualization
    if len(matches) > max_matches:
        sampled_matches = random.sample(matches, max_matches)
    else:
        sampled_matches = matches
    
    # Draw matches
    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, sampled_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Resize for display
    display_img = cv2.resize(match_img, (1280, 720))
    
    # Show result
    cv2.imshow("Matched Correspondences", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return display_img

def create_pointcloud(keypoints1, keypoints2, colors, camera_matrix1, camera_matrix2, R, T):
    """
    Create a colored point cloud from matched correspondences and camera parameters.
    """
    points_3d = []
    point_colors = []
    
    for i in range(len(keypoints1)):
        # Get 2D points
        pt1 = np.array([keypoints1[i].pt[0], keypoints1[i].pt[1], 1.0])
        pt2 = np.array([keypoints2[i].pt[0], keypoints2[i].pt[1], 1.0])
        
        # Triangulate
        point = triangulate_point(pt1, pt2, camera_matrix1, camera_matrix2, R, T)
        points_3d.append(point[:3])
        
        # Add color (BGR to RGB)
        color = colors[i][::-1] / 255.0  # Normalize to [0, 1] and convert BGR to RGB
        point_colors.append(color)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points_3d))
    pcd.colors = o3d.utility.Vector3dVector(np.array(point_colors))
    
    return pcd

def triangulate_point(pt1, pt2, K1, K2, R, T):
    """
    Triangulate a 3D point from corresponding 2D points in two views.
    """
    # Create projection matrices
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T))
    
    # Normalize points
    pt1_norm = np.linalg.inv(K1) @ pt1
    pt2_norm = np.linalg.inv(K2) @ pt2
    
    # DLT algorithm
    A = np.zeros((4, 4))
    A[0] = pt1_norm[0] * P1[2] - P1[0]
    A[1] = pt1_norm[1] * P1[2] - P1[1]
    A[2] = pt2_norm[0] * P2[2] - P2[0]
    A[3] = pt2_norm[1] * P2[2] - P2[1]
    
    # SVD solution
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    # Homogeneous to Euclidean
    X = X / X[3]
    
    return X

if __name__ == "__main__":
    # Parameters
    threshold = 40  # Adjust based on lighting conditions
    bit_depth = 10
    
    # Load images for view 1
    print("Loading images from view 1...")
    h_patterns_1 = sorted(glob.glob('../Data/GrayCodes/view1/*.jpg'))
    v_patterns_1 = sorted(glob.glob('../Data/GrayCodes/view1/*.jpg'))
    white_1 = h_patterns_1[0]
    black_1 = h_patterns_1[1]
    
    # Load images for view 2
    print("Loading images from view 0...")
    h_patterns_2 = sorted(glob.glob('../Data/GrayCodes/view0/*.jpg'))
    v_patterns_2 = sorted(glob.glob('../Data/GrayCodes/view0/*.jpg'))
    white_2 = h_patterns_2[0]
    black_2 = h_patterns_2[1]
    
    # Load all images
    h_images_1 = [cv2.imread(f) for f in h_patterns_1]
    v_images_1 = [cv2.imread(f) for f in v_patterns_1]
    h_images_2 = [cv2.imread(f) for f in h_patterns_2]
    v_images_2 = [cv2.imread(f) for f in v_patterns_2]
    
    # Decode patterns
    print("Decoding patterns for view 1...")
    result1, h_code1, v_code1, mask1 = decode_gray_patterns(
        h_images_1, v_images_1, white_1, black_1, threshold
    )
    
    print("Decoding patterns for view 2...")
    result2, h_code2, v_code2, mask2 = decode_gray_patterns(
        h_images_2, v_images_2, white_2, black_2, threshold
    )
    
    print(f"View 1: {len(result1)} valid pixels")
    print(f"View 2: {len(result2)} valid pixels")
    
    # Find correspondences
    keypoints1, keypoints2, matches, colors = find_correspondences(result1, result2)
    
    # Visualize matches
    match_img = draw_matched_correspondences(white_1, white_2, keypoints1, keypoints2, matches)
    
    # Save result
    cv2.imwrite("matched_correspondences.png", match_img)
    
    print("Done. Visualized matches saved to 'matched_correspondences.png'")