import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import GrayCodeEncoder
import random
import open3d as o3d


def get_diff(images, treshold=30):
    """
    Get the difference between the images and create a bitmask based on the treshold.

    :param images: List of images.
    :param treshold: Treshold for the difference between the images.
    :return: List of bitmasks.
    """
    bitmasks = []
    for i in range(0, len(images), 2):
        # img0 is the image with a gray code pattern and img1 is the image with the inverse gray code pattern
        img0 = images[i]
        img1 = images[i+1]

        # Get the difference between the images
        diff = cv2.subtract(img1, img0)

        # Create a bitmask based on the treshold
        bitmask = np.zeros_like(diff, dtype=np.uint8)
        bitmask[diff > treshold] = 1
        bitmask[diff <= treshold] = 0

        bitmasks.append(bitmask)

    return bitmasks

def validate_pixels(img0, img1, treshhold=50):
    """
    Validate pixels in the images based on the difference between them.
    """
    # Get the difference between the images
    diff = cv2.absdiff(img0, img1)
    mask = np.zeros_like(diff, dtype=np.uint8)
    mask[diff > treshhold] = 1 # Set to 1 if difference is larger than treshhold
    return mask

def decode_gray_pattern(images, treshhold=30):
    """
    Decode the gray code patterns from the captured images.

    :param images: List of images containing the gray code patterns.
    :param treshhold: Treshold for the difference between the images.
    :return: List of (x, y, identifier) for each pixel in the images.
    """
    # Validate pixels
    valid_mask = validate_pixels(images[0], images[1])

    height, width = images[0].shape
    identifier = np.zeros((height, width), dtype=np.uint32)
    
    # Get the difference between the images
    masks = get_diff(images, treshhold)

    # Create an identifier for each pixel
    for i in range(len(masks)):
        identifier |= (masks[i].astype(np.uint32) << i)

    # Apply valid mask to identifier
    identifier[valid_mask == 0] = 0

    # Create a list of (x, y, identifier) for valid pixels
    identifier_list = [(x, y, identifier[y, x]) for y in range(height) for x in range(width) if valid_mask[y, x] == 1]

    return identifier_list

def find_correspondences(identifier_list1, identifier_list2):
    """
    Search for corresponding pixels between two camera viewpoints based on identifiers.

    :param identifier_list1: List with (x, y, identifier) for camera 1.
    :param identifier_list2: List with (x, y, identifier) for camera 2.
    :return: List with corresponding pixel pairs [(pt1, pt2)]
    """
    # Make a dictionary of identifiers for camera 2
    id_dict = {identifier: (x, y) for x, y, identifier in identifier_list2}

    matches = []
    keypoints1, keypoints2 = [], []

    # Search for corresponding identifiers in camera 1 and 2
    for x1, y1, identifier in identifier_list1:
        if identifier in id_dict:  # Check if identifier exists in camera 2
            x2, y2 = id_dict[identifier]
            #(x1, y1, x2, y2)
            keypoints1.append(cv2.KeyPoint(x1, y1, 1))  # OpenCV KeyPoint needed for drawMatches
            keypoints2.append(cv2.KeyPoint(x2, y2, 1))
            matches.append(cv2.DMatch(len(matches), len(matches), 0))  # Match object with ID's

    return keypoints1, keypoints2, matches

def draw_matched_correspondences(img1, img2, keypoints1, keypoints2, matches, max_matches=50):
    """
    Draw matches between two camera images using OpenCV's drawMatches.
    
    :param img1: First camera image.
    :param img2: Second camera image.
    :param keypoints1: Keypoints from first camera.
    :param keypoints2: Keypoints from second camera.
    :param matches: List of cv2.DMatch objects.
    :param max_matches: Maximum number of matches to visualize.
    """

    # Take a random sample of matches to draw to avoid clutter
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

    # Load and resize images in grayscale
    images0 = []
    images1 = []
    for i in range(len(images_view0)):
        resized0 = cv2.resize(cv2.imread(images_view0[i], cv2.IMREAD_GRAYSCALE), (1920, 1080))
        resized1 = cv2.resize(cv2.imread(images_view1[i], cv2.IMREAD_GRAYSCALE), (1920, 1080))
        images0.append(resized0)
        images1.append(resized1)

    # Decode the gray code patterns
    result0 = decode_gray_pattern(images0)
    result1 = decode_gray_pattern(images1)

    img1 = images0[0]
    img2 = images1[0]

    # Find corresponding pixels
    keypoints1, keypoints2, matches = find_correspondences(result0, result1)

    # Draw matched correspondences
    draw_matched_correspondences(img1, img2, keypoints1, keypoints2, matches)

    