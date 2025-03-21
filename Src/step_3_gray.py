import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import GrayCodeEncoder
import random
import open3d as o3d

#import step_2.py file from same folder
from step_2_gray import get_diff, validate_pixels, decode_gray_pattern, find_correspondences, draw_matched_correspondences

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

def close_visualizer(vis):
    vis.close()  # This will close the Open3D window
    return False  # Returning False ensures it stops updating

def visualize_cameras_axises(R, T):
    # Visualiseer camrera met 3 axissen
    # Maak de Open3D Visualizer 
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # Eerste camera (in de oorsprong)
    camera1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(camera1)

    # Tweede camera met de berekende rotatie en translatie
    camera2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    transformation = np.eye(4)
    transformation[:3, :3] = R  # Rotatie
    transformation[:3, 3] = T.T  # Translatie
    camera2.transform(transformation)

    # Voeg de tweede camera toe aan de visualizer
    vis.add_geometry(camera2)

    vis.register_key_callback(ord("C"), close_visualizer)

    # Start de visualisatie
    vis.run()
    vis.destroy_window()

def visualize_cameras_pyramids(R, T):
    # Visualiseer camrera met pyramides
    # Visualiseer camera standpunten met Open3D
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # Eerste camera in de oorsprong (gewoon een piramide)
    camera1 = create_camera_pyramid()
    vis.add_geometry(camera1)

    # Tweede camera transformeren met rotatie en translatie
    camera2 = create_camera_pyramid()

    # Combineer rotatie (R) en translatie (T) in een 4x4 transformatie matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T.T

    # Pas de transformatie toe op de piramide voor de tweede camera
    camera2.transform(transform)
    vis.add_geometry(camera2)

    vis.register_key_callback(ord("C"), close_visualizer)

    # Start de visualisatie
    vis.run()
    vis.destroy_window()

def triangulate_opencv(pts1, pts2, K, R, T):
    """
    Bereken 3D punten met OpenCV's triangulatiefunctie.
    
    :param pts1: 2D punten in camera 1 (Nx2 array)
    :param pts2: 2D punten in camera 2 (Nx2 array)
    :param K: Camera-intrinsieken (3x3 matrix)
    :param R: Rotatie tussen de camera’s (3x3 matrix)
    :param T: Translatie tussen de camera’s (3x1 vector)
    :return: 3D punten in niet-homogene coördinaten
    """
    # Projectiematrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Eerste camera op oorsprong
    P2 = K @ np.hstack((R, T))  # Tweede camera met R en T

    # Zet 2D punten om naar homogene coördinaten
    pts1_h = np.array(pts1).T
    pts2_h = np.array(pts2).T

    # Triangulatie met OpenCV
    points_4D = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])

    # Omzetten naar niet-homogene coördinaten
    points_3D = points_4D[:3] / points_4D[3]

    return points_3D.T  # Transponeren naar Nx3 vorm

def triangulate_manual(pts1, pts2, K, R, T):
    """
    Bereken 3D punten door het oplossen van Ax=0 met SVD.

    :param pts1: 2D punten in camera 1 (Nx2 array)
    :param pts2: 2D punten in camera 2 (Nx2 array)
    :param K: Camera-intrinsieken (3x3 matrix)
    :param R: Rotatie tussen de camera’s (3x3 matrix)
    :param T: Translatie tussen de camera’s (3x1 vector)
    :return: 3D punten in niet-homogene coördinaten
    """
    # Projectiematrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, T))

    points_3D = []

    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        # Rijvergelijkingen opstellen voor matrix A
        A = np.array([
            x1 * P1[2, :] - P1[0, :],  # x1 * P1_row3 - P1_row1
            y1 * P1[2, :] - P1[1, :],  # y1 * P1_row3 - P1_row2
            x2 * P2[2, :] - P2[0, :],  # x2 * P2_row3 - P2_row1
            y2 * P2[2, :] - P2[1, :]   # y2 * P2_row3 - P2_row2
        ])

        # SVD oplossen: laatste kolom van V geeft X
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :]

        # Dehomogeniseer
        X = X[:3] / X[3]
        points_3D.append(X)

    return np.array(points_3D)

def visualize_3D_points(points_3D):
    """
    Visualiseer 3D punten met Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D)

    # Maak de punten zichtbaar als witte bolletjes
    pcd.paint_uniform_color([1, 0, 0])

    # Visualisatie starten
    o3d.visualization.draw_geometries([pcd])


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

# draw_matched_correspondences(img1, img2, keypoints1, keypoints2, matches)

K = np.array([[9.51663140e+03, 0.00000000e+00, 2.81762458e+03],[0.00000000e+00, 8.86527952e+03, 1.14812762e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

E, mask, pts1, pts2 = compute_essential_matrix(keypoints1, keypoints2, matches, K)
print(E)

# R = rotatie vector
# T = translatie vector
_, R, T, mask = cv2.recoverPose(E, pts1, pts2, K)
print(R)
print(T)

visualize_cameras_axises(R, T)
visualize_cameras_pyramids(R, T)

triangulate_opencv(pts1, pts2, K, R, T)

points_3D_opencv = triangulate_opencv(pts1, pts2, K, R, T)
points_3D_manual = triangulate_manual(pts1, pts2, K, R, T)

print("3D punten (OpenCV triangulatePoints):\n", points_3D_opencv)
print("3D punten (Manuele SVD triangulatie):\n", points_3D_manual)

# 3D punten visualiseren
visualize_3D_points(points_3D_opencv)