import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import GrayCodeEncoder
import random
import open3d as o3d
import sys


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

    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, sampled_matches, img1.shape,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    match_img = cv2.resize(match_img, (1280, 720))
    cv2.imshow("Matched Correspondences", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_gray_code_patterns():
    """
    Generate gray code patterns and save them to disk.
    """
    gce = GrayCodeEncoder.GrayCodeEncoder(1080, 1920, 10)
    for i in range(0, 40):
        cv2.imwrite(f"Result/gce_patterns/{i:02d}.jpg", gce.get_encoding_pattern(i))


def get_image_paths(number_of_cameras):
    """
    Get the image paths for the specified number of cameras.
    
    :param number_of_cameras: Number of cameras to get image paths for.
    :return: List of image paths.
    """
    images = []
    for i in range(number_of_cameras):
        images.append(glob.glob(f'../Data/GrayCodes/view{i}/*.jpg'))

    return images


def read_images(image_paths):
    """
    Read images from the specified paths and return them as a list.
    
    :param image_paths: List of image paths.
    :return: List of images.
    """
    images = []
    for camera in image_paths:
        images.append([])
        for image in camera:
            img = cv2.resize(cv2.imread(image, cv2.IMREAD_GRAYSCALE), (1920, 1080))
            if img is not None:
                images[len(images) - 1].append(img)
    return images



def decode_gray_pattern_n_cameras(images):
    """
    Decode gray code patterns for multiple cameras.

    :param images: List of lists of images for each camera.
    :return: List of decoded identifiers for each camera.
    """
    results = []
    for camera_images in images:
        result = decode_gray_pattern(camera_images)
        results.append(result)
    return results


def find_correspondences_n_cameras(results):
    """
    Find correspondences between multiple cameras based on decoded identifiers.

    :param results: List of decoded identifiers for each camera.
    :return: List of keypoints and matches for each pair of cameras.
    """
    correspondences = []
    for i in range(len(results) - 1):
        for j in range(i + 1, len(results)):
            keypoints1, keypoints2, matches = find_correspondences(results[i], results[j])
            correspondences.append((keypoints1, keypoints2, matches))
    return correspondences

if __name__ == "__main__":

    number_of_cameras = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    show_matches = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

    generate_gray_code_patterns()

    image_paths = get_image_paths(number_of_cameras)
    loaded_images = read_images(image_paths)
    print(f"Number of cameras: {len(loaded_images)}")
    print(f"Number of images per camera: {len(loaded_images[0])}")

    # Decode the gray codes
    results = decode_gray_pattern_n_cameras(loaded_images)

    # Find corresponding pixels
    correspondences = find_correspondences_n_cameras(results)

    img1 = loaded_images[0][0]
    img2 = loaded_images[1][0]

    # Draw matched correspondences
    for i in correspondences:
        keypoints1, keypoints2, matches = i
        draw_matched_correspondences(img1, img2, keypoints1, keypoints2, matches)

    