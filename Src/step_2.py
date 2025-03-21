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

import cv2
import numpy as np

def compute_essential_matrix(keypoints1, keypoints2, matches, K):
    """
    Bereken de essentiële matrix uit corresponderende punten.
    
    :param keypoints1: OpenCV keypoints in camera 1
    :param keypoints2: OpenCV keypoints in camera 2
    :param matches: Lijst van overeenkomende punten (cv2.DMatch)
    :param K: Camera-calibratiematrix (3x3 numpy array)
    :return: Essentiële matrix E
    """
    # Zet matches om in pixelcoördinaten
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Bereken de essentiële matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    return E, mask, pts1, pts2

def create_camera_pyramid(size=0.1):
        # Vertices voor een piramide (basis + top)
        vertices = np.array([
            [0, 0, 0],  # Basis 1 (Oorsprong)
            [size, size, size],  # Basis 2
            [-size, size, size],  # Basis 3
            [-size, -size, size],  # Basis 4
            [size, -size, size],  # Basis 5
            [0, 0, 2*size],  # Top (punt)
        ])
    
        # Vlakken voor de piramide (4 zijvlakken + basis)
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],  # Zijvlakken
            [1, 2, 3], [1, 3, 4], [1, 4, 2]  # Basisvlakken
        ])
        
        # Maak de mesh
        pyramid = o3d.geometry.TriangleMesh()
        pyramid.vertices = o3d.utility.Vector3dVector(vertices)
        pyramid.triangles = o3d.utility.Vector3iVector(faces)
        
        # Stel kleuren in (rood voor de camera)
        pyramid.paint_uniform_color([0.9, 0.2, 0.2])  # Rood kleur
    
        return pyramid

if __name__ == "__main__":
    gce = GrayCodeEncoder.GrayCodeEncoder(1920, 1080, 10)

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

    keypoints1, keypoints2, matches = find_correspondences(result0, result1)

    draw_matched_correspondences(img1, img2, keypoints1, keypoints2, matches)

    # K = np.array([[9.51663140e+03, 0.00000000e+00, 2.81762458e+03],[0.00000000e+00, 8.86527952e+03, 1.14812762e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    # E, mask, pts1, pts2 = compute_essential_matrix(keypoints1, keypoints2, matches, K)
    # print(E)

    # # R = rotatie vector
    # # T = translatie vector
    # _, R, T, mask = cv2.recoverPose(E, pts1, pts2, K)
    # print(R)
    # print(T)

    # import open3d as o3d

    # # Maak de Open3D Visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # # Eerste camera (in de oorsprong)
    # camera1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # vis.add_geometry(camera1)

    # # Tweede camera met de berekende rotatie en translatie
    # camera2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # transformation = np.eye(4)
    # transformation[:3, :3] = R  # Rotatie
    # transformation[:3, 3] = T.T  # Translatie
    # camera2.transform(transformation)

    # # Voeg de tweede camera toe aan de visualizer
    # vis.add_geometry(camera2)

    # # Start de visualisatie
    # vis.run()
    # vis.destroy_window()

    # # Visualiseer camera standpunten met Open3D
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # # Eerste camera in de oorsprong (gewoon een piramide)
    # camera1 = create_camera_pyramid()
    # vis.add_geometry(camera1)

    # # Tweede camera transformeren met rotatie en translatie
    # camera2 = create_camera_pyramid()

    # # Combineer rotatie (R) en translatie (T) in een 4x4 transformatie matrix
    # transform = np.eye(4)
    # transform[:3, :3] = R
    # transform[:3, 3] = T.T

    # # Pas de transformatie toe op de piramide voor de tweede camera
    # camera2.transform(transform)
    # vis.add_geometry(camera2)

    # # Start de visualisatie
    # vis.run()
    # vis.destroy_window()