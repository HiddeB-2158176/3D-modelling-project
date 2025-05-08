import cv2
import glob
import numpy as np
import GrayCodeEncoder
import random
import open3d as o3d


def get_diff(images, threshold=30):
    bitmasks = []
    for i in range(0, len(images), 2):
        img0 = images[i]
        img1 = images[i+1]
        diff = cv2.subtract(img1, img0)
        bitmask = np.zeros_like(diff, dtype=np.uint8)
        bitmask[diff > threshold] = 1
        bitmask[diff <= threshold] = 0
        bitmasks.append(bitmask)
    return bitmasks

def validate_pixels(img0, img1, threshold=50):
    diff = cv2.absdiff(img0, img1)
    mask = np.zeros_like(diff, dtype=np.uint8)
    mask[diff > threshold] = 1
    return mask

def decode_gray_pattern(images, threshold=30):
    valid_mask = validate_pixels(images[0], images[1])
    height, width = images[0].shape
    identifier = np.zeros((height, width), dtype=np.uint32)
    masks = get_diff(images, threshold)

    for i in range(len(masks)):
        identifier |= (masks[i].astype(np.uint32) << i)

    identifier[valid_mask == 0] = 0
    identifier_list = [(x, y, identifier[y, x]) for y in range(height) for x in range(width) if valid_mask[y, x] == 1]
    return identifier_list

def load_camera_images(base_path, num_cameras, image_size):
    all_images = []
    for cam_idx in range(num_cameras):
        view_paths = sorted(glob.glob(f'{base_path}/view{cam_idx}/*.png'))
        images = [cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), image_size) for p in view_paths]
        all_images.append(images)
    return all_images

def decode_all_views(all_images, threshold=30):
    return [decode_gray_pattern(images, threshold) for images in all_images]

def find_multi_view_correspondences(identifier_lists):
    correspondences = {}
    for cam_idx, id_list in enumerate(identifier_lists):
        for x, y, ident in id_list:
            if ident == 0:
                continue
            if ident not in correspondences:
                correspondences[ident] = []
            correspondences[ident].append((cam_idx, x, y))
    return {k: v for k, v in correspondences.items() if len(v) >= 2}

def get_keypoints_and_matches(correspondences, view1_idx, view2_idx):
    keypoints1 = []
    keypoints2 = []
    matches = []
    idx = 0

    for ident, views in correspondences.items():
        pt1 = [v for v in views if v[0] == view1_idx]
        pt2 = [v for v in views if v[0] == view2_idx]
        if pt1 and pt2:
            _, x1, y1 = pt1[0]
            _, x2, y2 = pt2[0]
            keypoints1.append(cv2.KeyPoint(x1, y1, 1))
            keypoints2.append(cv2.KeyPoint(x2, y2, 1))
            matches.append(cv2.DMatch(idx, idx, 0))
            idx += 1

    return keypoints1, keypoints2, matches

def visualize_correspondences(all_images, correspondences):
    first_images = [cam_images[40] for cam_images in all_images]
    num_views = len(first_images)

    if num_views == 3:
        # Create composite image
        stacked_img = np.hstack(first_images)
        offset = [0, first_images[0].shape[1], first_images[0].shape[1] + first_images[1].shape[1]]
        canvas = cv2.cvtColor(stacked_img, cv2.COLOR_GRAY2BGR)

        # Draw lines between matched points
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for ident, views in correspondences.items():
            if len(views) < 2:
                continue
            pts = {v[0]: (v[1], v[2]) for v in views}
            if all(v in pts for v in [0, 1]):
                pt1 = (pts[0][0] + offset[0], pts[0][1])
                pt2 = (pts[1][0] + offset[1], pts[1][1])
                cv2.line(canvas, pt1, pt2, colors[0], 1)
            if all(v in pts for v in [1, 2]):
                pt1 = (pts[1][0] + offset[1], pts[1][1])
                pt2 = (pts[2][0] + offset[2], pts[2][1])
                cv2.line(canvas, pt1, pt2, colors[1], 1)
            if all(v in pts for v in [0, 2]):
                pt1 = (pts[0][0] + offset[0], pts[0][1])
                pt2 = (pts[2][0] + offset[2], pts[2][1])
                cv2.line(canvas, pt1, pt2, colors[2], 1)

        canvas = cv2.resize(canvas, (1600, 600))
        cv2.imshow("3-View Correspondences", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        for i in range(num_views):
            for j in range(i+1, num_views):
                kps1, kps2, matches = get_keypoints_and_matches(correspondences, i, j)
                img1 = first_images[i]
                img2 = first_images[j]

                # Convert to color for drawMatches
                max_matches=50
                sampled_matches = random.sample(matches, min(len(matches), max_matches))
                img_matches = cv2.drawMatches(img1, kps1, img2, kps2, matches, None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                img_matches = cv2.resize(img_matches, (1600, 600))
                cv2.imshow(f"Matches View {i} - View {j}", img_matches)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    IMAGE_SIZE = (1920, 1080)
    NUM_CAMERAS = 3
    THRESHOLD = 30
    base_path = "../Result/ownDataset1/"

    K = [[644.94563526, 0, 641.4013732 ],[0, 644.91566198, 410.25960708],[0, 0, 1]]

    # Load and decode images
    all_images = load_camera_images(base_path, NUM_CAMERAS, IMAGE_SIZE)
    print(f"Loaded {len(all_images)} cameras with {len(all_images[0])} images each.")
    decoded_identifiers = decode_all_views(all_images, threshold=THRESHOLD)

    # Find matches between views
    correspondences = find_multi_view_correspondences(decoded_identifiers)

    visualize_correspondences(all_images, correspondences)
