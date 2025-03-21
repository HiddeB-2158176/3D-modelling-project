import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import GrayCodeEncoder
import random
import open3d as o3d


def get_diff(images, threshold=30):
    """
    Compute bitmask differences while preserving color information.
    """
    bitmask_list = []
    color_list = []
    
    for i in range(0, len(images), 2):
        img0 = images[i]
        img1 = images[i+1]

        # Compute per-channel absolute difference
        diff = cv2.absdiff(img1, img0)
        bitmask = np.zeros_like(diff, dtype=np.uint8)
        bitmask[np.linalg.norm(diff, axis=2) > threshold] = 1
        bitmask[np.linalg.norm(diff, axis=2) <= threshold] = 0

        bitmask_list.append(bitmask)

        # Store color of brighter image (img1) for correct color association
        color_list.append(img1)

    return bitmask_list, color_list

def validate_pixels(img0, img1, threshold=50):
    """
    Validate pixels based on intensity difference.
    """
    diff = cv2.absdiff(img0, img1)
    mask = np.zeros(diff.shape[:2], dtype=np.uint8)
    mask[np.linalg.norm(diff, axis=2) > threshold] = 1  # Apply norm to RGB channels
    return mask

def decode_gray_pattern(images, threshold=30):
    """
    Decode Gray code pattern and retain color information.
    """
    valid_mask = validate_pixels(images[0], images[1])
    height, width, _ = images[0].shape
    identifier = np.zeros((height, width), dtype=np.uint32)

    masks, colors = get_diff(images, threshold)

    for i in range(len(masks)):
        identifier |= (masks[i][:, :, 1].astype(np.uint32) << i)  # Use red channel as reference

    identifier[valid_mask == 0] = 0
    # for y in range(height): 
    #     for x in range(width):
    #         if valid_mask[y, x] == 1:
    #             print(x, y, identifier[y, x])

    # Store 3D points with color
    identifier_list = [
        (x, y, identifier[y, x], colors[-1][y, x])  # Store RGB color from last captured image
        for y in range(height) for x in range(width) if valid_mask[y, x] == 1
    ]

    return identifier_list

def find_correspondences(identifier_list1, identifier_list2):
    """
    Find pixel correspondences and retain color information.
    """
    id_dict = {identifier: (x, y, color) for x, y, identifier, color in identifier_list2}

    matches = []
    keypoints1, keypoints2, colors = [], [], []

    for x1, y1, identifier, color1 in identifier_list1:
        if identifier in id_dict:
            x2, y2, color2 = id_dict[identifier]
            print(x1, y1, x2, y2)
            keypoints1.append(cv2.KeyPoint(x1, y1, 1))
            keypoints2.append(cv2.KeyPoint(x2, y2, 1))
            matches.append(cv2.DMatch(len(matches), len(matches), 0))
            colors.append(color1)  # Use color from first camera

    return keypoints1, keypoints2, matches, colors

def draw_matched_correspondences(img1, img2, keypoints1, keypoints2, matches, max_matches=50):
    """
    Draw correspondences with OpenCV.
    """
    sampled_matches = random.sample(matches, min(len(matches), max_matches))
    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, sampled_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    match_img = cv2.resize(match_img, (1280, 720))
    cv2.imshow("Matched Correspondences", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gce = GrayCodeEncoder.GrayCodeEncoder(1920, 1080, 10)

    images_view0 = glob.glob('../Data/GrayCodes/view0/*.jpg')
    images_view1 = glob.glob('../Data/GrayCodes/view1/*.jpg')

    images0 = []
    images1 = []
    for i in range(len(images_view0)):
        resized0 = cv2.resize(cv2.imread(images_view0[i], cv2.IMREAD_COLOR), (1920, 1080))
        resized1 = cv2.resize(cv2.imread(images_view1[i], cv2.IMREAD_COLOR), (1920, 1080))
        images0.append(resized0)
        images1.append(resized1)

    result0 = decode_gray_pattern(images0)
    result1 = decode_gray_pattern(images1)
    print(result0[:5])
    print(result1[:5])

    img1 = images0[0]
    img2 = images1[0]

    keypoints1, keypoints2, matches, colors = find_correspondences(result0, result1)

    draw_matched_correspondences(img1, img2, keypoints1, keypoints2, matches)

    