import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import GrayCodeEncoder
import random


def get_diff(images, treshold=30):
    arr = []
    for i in range(0, len(images), 2):
        img0 = images[i]
        img1 = images[i+1]

        diff = cv2.subtract(img1, img0)
        bitmask = np.zeros_like(diff, dtype=np.uint8)
        bitmask[diff > treshold] = 1
        bitmask[diff <= treshold] = 0

        # diff = cv2.resize(diff, (960, 540))
        arr.append(bitmask)
        # cv2.imshow("diff", diff)
        # cv2.waitKey(0)
    return arr

def validate_pixels(img0, img1, treshhold=50):
    diff = cv2.absdiff(img0, img1)
    mask = np.zeros_like(diff, dtype=np.uint8)
    mask[diff > treshhold] = 1
    return mask

def decode_gray_pattern(images, treshhold=30):
    valid_mask = validate_pixels(images[0], images[1])
    height, width = images[0].shape
    identifier = np.zeros((height, width), dtype=np.uint32)
    
    masks = get_diff(images, treshhold)
    for i in range(len(masks)):
        identifier |= (masks[i].astype(np.uint32) << i)

    identifier[valid_mask == 0] = 0

    identifier_list = [(x, y, identifier[y, x]) for y in range(height) for x in range(width) if valid_mask[y, x] == 1]
    return identifier_list

def find_correspondences(identifier_list1, identifier_list2):
    """
    Zoek overeenkomstige pixels tussen twee camerastandpunten op basis van identifiers.
    
    :param identifier_list1: Lijst met (x, y, identifier) voor camera 1.
    :param identifier_list2: Lijst met (x, y, identifier) voor camera 2.
    :return: Lijst met overeenkomstige pixelparen [(pt1, pt2)]
    """
    # Maak een dictionary voor snelle lookup van camera 2 identifiers
    id_dict = {identifier: (x, y) for x, y, identifier in identifier_list2}

    matches = []
    keypoints1, keypoints2 = [], []

    for x1, y1, identifier in identifier_list1:
        if identifier in id_dict:  # Check of identifier in de andere camera zit
            x2, y2 = id_dict[identifier]
            keypoints1.append(cv2.KeyPoint(x1, y1, 1))  # OpenCV KeyPoint nodig voor drawMatches
            keypoints2.append(cv2.KeyPoint(x2, y2, 1))
            matches.append(cv2.DMatch(len(matches), len(matches), 0))  # Match object met ID's

    return keypoints1, keypoints2, matches

def draw_matched_correspondences(img1, img2, keypoints1, keypoints2, matches, max_matches=50):
    """
    Teken overeenkomsten tussen twee camerabeelden met OpenCV's drawMatches.
    
    :param img1: Eerste camerabeeld.
    :param img2: Tweede camerabeeld.
    :param keypoints1: Keypoints uit eerste camera.
    :param keypoints2: Keypoints uit tweede camera.
    :param matches: Lijst van cv2.DMatch objecten.
    :param max_matches: Maximum aantal matches om te visualiseren.
    """
    # Neem een random subset van de matches om het overzichtelijk te houden
    sampled_matches = random.sample(matches, min(len(matches), max_matches))

    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, sampled_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    match_img = cv2.resize(match_img, (1920, 1080))
    cv2.imshow("Matched Correspondences", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

gce = GrayCodeEncoder.GrayCodeEncoder(1920, 1080, 10)

# for i in range(len(gce.patterns)):
#     cv2.imshow("image", gce.get_encoding_pattern(i))
#     cv2.waitKey(0)

images_view0 = glob.glob('../Data/GrayCodes/view0/*.jpg')
images_view1 = glob.glob('../Data/GrayCodes/view1/*.jpg')

# diff0 = get_diff(images_view0)
# diff1 = get_diff(images_view1)

# print(len(diff0), len(diff1))

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

keypoints1, keypoints2, matches = find_correspondences(result0, result1)

draw_matched_correspondences(img1, img2, keypoints1, keypoints2, matches)