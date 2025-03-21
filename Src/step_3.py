import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import GrayCodeEncoder
import random
import open3d as o3d

#import step_2.py file from same folder
from step_2 import get_diff, validate_pixels, decode_gray_pattern, find_correspondences, draw_matched_correspondences

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

def close_visualizer(vis):
    vis.close()  # This will close the Open3D window
    return False  # Returning False ensures it stops updating

# Voeg de tweede camera toe aan de visualizer
vis.add_geometry(camera2)

vis.register_key_callback(ord("C"), close_visualizer)

# Start de visualisatie
vis.run()
vis.destroy_window()

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