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
    Calculate the essential matrix from corresponding points.
    
    :param keypoints1: OpenCV keypoints in camera 1
    :param keypoints2: OpenCV keypoints in camera 2
    :param matches: Lijst van overeenkomende punten (cv2.DMatch)
    :param matches: List of matching points (cv2.DMatch)
    :param K: Camera calibration matrix (3x3 numpy array)
    :return: Essential matrix E
    """

    # Convert matches to pixel coordinates
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Calculate the essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    return E, mask, pts1, pts2

def create_camera_pyramid(size=0.1):
    """
    Create a pyramid mesh representing a camera.
    """
    # Vertices for a pyramid (base + top)
    vertices = np.array([
        [0, 0, 0], 
        [size, size, size], 
        [-size, size, size], 
        [-size, -size, size],  
        [size, -size, size], 
        [0, 0, 2*size], 
    ])

    # Planes for the pyramid (4 side planes + base)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], 
        [1, 2, 3], [1, 3, 4], [1, 4, 2] 
    ])
    
    pyramid = o3d.geometry.TriangleMesh()
    pyramid.vertices = o3d.utility.Vector3dVector(vertices)
    pyramid.triangles = o3d.utility.Vector3iVector(faces)
    
    pyramid.paint_uniform_color([0.9, 0.2, 0.2])  # Red color

    return pyramid

def close_visualizer(vis):
    """
    Close the Open3D visualizer.
    """
    vis.close()  # This will close the Open3D window
    return False  # Returning False ensures it stops updating

def visualize_cameras_axises(R, T):
    """
    Visualize the cameras with axises.
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # First camera in the origin
    camera1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(camera1)

    # Second camera with the calculated rotation and translation
    camera2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    transformation = np.eye(4)
    transformation[:3, :3] = R  # Rotatie
    transformation[:3, 3] = T.T  # Translatie
    camera2.transform(transformation)

    # Add the second camera to the visualizer
    vis.add_geometry(camera2)

    vis.register_key_callback(ord("C"), close_visualizer)

    vis.run()
    vis.destroy_window()

def visualize_cameras_pyramids(R, T):
    """
    Visualize the cameras with pyramids.
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # First camera in the origin
    camera1 = create_camera_pyramid()
    vis.add_geometry(camera1)

    # Second camera with the calculated rotation and translation
    camera2 = create_camera_pyramid()

    # Combine the rotation and translation into a single transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T.T

    # Apply the transformation to the pyramid for the second camera
    camera2.transform(transform)
    vis.add_geometry(camera2)

    vis.register_key_callback(ord("C"), close_visualizer)

    vis.run()
    vis.destroy_window()

def triangulate_opencv(pts1, pts2, K, R, T):
    """
    Calculate 3D points with OpenCV's triangulation function.
    
    :param pts1: 2D points from camera 1 (Nx2 array)
    :param pts2: 2D points from camera 2 (Nx2 array)
    :param K: Camera intrinsics (3x3 matrix)
    :param R: Rotation between the cameras (3x3 matrix)
    :param T: Translation between the cameras (3x1 vector)
    :return: 3D points in non-homogeneous coordinates
    """
    # Projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Eerste camera op oorsprong
    P2 = K @ np.hstack((R, T))  # Tweede camera met R en T

    # Convert 2D points to homogeneous coordinates
    pts1_h = np.array(pts1).T
    pts2_h = np.array(pts2).T

    # Triangulate points with OpenCV
    points_4D = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])

    # Converte to non-homogeneous 3D coordinates
    points_3D = points_4D[:3] / points_4D[3]

    return points_3D.T  # Transpone the matrix to get Nx3 array

def triangulate_manual(pts1, pts2, K, R, T):
    """
    Calculate 3D points by solving Ax=0 with SVD.

    :param pts1: 2D points from camera 1 (Nx2 array)
    :param pts2: 2D points from camera 2 (Nx2 array)
    :param K: Camera intrinsics (3x3 matrix)
    :param R: Rotation between the cameras (3x3 matrix)
    :param T: Translation between the cameras (3x1 vector)
    :return: 3D points in non-homogeneous coordinates
    """

    # Projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, T))

    points_3D = []

    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        # The formula Ax=0 is used to solve for 3D points (see extra info on triangulation)
        A = np.array([
            x1 * P1[2, :] - P1[0, :],  # x1 * P1_row3 - P1_row1
            y1 * P1[2, :] - P1[1, :],  # y1 * P1_row3 - P1_row2
            x2 * P2[2, :] - P2[0, :],  # x2 * P2_row3 - P2_row1
            y2 * P2[2, :] - P2[1, :]   # y2 * P2_row3 - P2_row2
        ])

        # # Solve for the 3D point using SVD (last column of Vt gives x)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :]

        # Dehomogenize the 3D point
        X = X[:3] / X[3]
        points_3D.append(X)

    return np.array(points_3D)

def visualize_3D_points(points_3D, colors):
    """
    Visualize the 3D pointcloud with color.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D)

    # Add colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors) 

    # Create a visualizer with a key callback
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.add_geometry(pcd)

    # Make it so you can close the window with the "C" key
    vis.register_key_callback(ord("C"), close_visualizer)

    vis.run()
    vis.destroy_window()

def get_colors(img1, img2, pts1, pts2):
    colors = []
    inifcount = 0
    inoutcount = 0
    for (x1, y1), (x2, y2) in zip(pts1.astype(int), pts2.astype(int)):
        # Ensure coordinates are within image bounds
        
        if 0 <= x1 < img1.shape[1] and 0 <= y1 < img1.shape[0] and \
           0 <= x2 < img2.shape[1] and 0 <= y2 < img2.shape[0]:

            # Extract colors as float32 to avoid overflow
            color1 = img1[y1, x1].astype(np.float32)  # From first image
            color2 = img2[y2, x2].astype(np.float32)  # From second image

            avg_color = (color1 + color2) / 2  # Average the colors
            inifcount = inifcount + 1
        else:
            inoutcount = inoutcount + 1
            avg_color = [128, 128, 128]  # Default gray for out-of-bounds points

        colors.append(avg_color)

    colors = np.array(colors, dtype=np.float32) / 255.0  # Normalize to range [0,1]
    return colors

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

img1Color = cv2.resize(cv2.imread(images_view0[0], cv2.IMREAD_COLOR), (1920, 1080))
img2Color = cv2.resize(cv2.imread(images_view1[0], cv2.IMREAD_COLOR), (1920, 1080))

keypoints1, keypoints2, matches = find_correspondences(result0, result1)

# draw_matched_correspondences(img1, img2, keypoints1, keypoints2, matches)

K = np.array([[9.51663140e+03, 0.00000000e+00, 2.81762458e+03],[0.00000000e+00, 8.86527952e+03, 1.14812762e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

E, mask, pts1, pts2 = compute_essential_matrix(keypoints1, keypoints2, matches, K)

# R = rotatie vector
# T = translatie vector
_, R, T, mask = cv2.recoverPose(E, pts1, pts2, K)

visualize_cameras_axises(R, T)
visualize_cameras_pyramids(R, T)

points_3D_opencv = triangulate_opencv(pts1, pts2, K, R, T)
points_3D_manual = triangulate_manual(pts1, pts2, K, R, T)

# cv2.imshow('img1', img1Color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('img2', img2Color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

colors = get_colors(img1Color, img2Color, pts1, pts2)

# 3D punten visualiseren
visualize_3D_points(points_3D_manual, colors)