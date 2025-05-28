import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import GrayCodeEncoder
import random
import open3d as o3d
import sys

#import step_2.py file from same folder
from step_2 import decode_gray_pattern, find_correspondences
from step_2 import get_image_paths, read_images, decode_gray_pattern_n_cameras, find_correspondences_n_cameras

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

    :param size: Size of the pyramid
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

    :param vis: Open3D visualizer object
    """
    vis.close()  # This will close the Open3D window
    return False  # Returning False ensures it stops updating

def visualize_cameras_axises(positions):
    """
    Visualize the cameras with axises.

    :param R: Rotation matrix
    :param T: Translation vector
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # First camera in the origin
    camera1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(camera1)

    for pos in positions:
        # Second camera with the calculated rotation and translation
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        transformation = np.eye(4)
        transformation[:3, :3] = pos[0]  # Rotatie
        transformation[:3, 3] = pos[1].T  # Translatie
        camera.transform(transformation)

        # Add the second camera to the visualizer
        vis.add_geometry(camera)

    vis.register_key_callback(ord("C"), close_visualizer)

    vis.run()
    vis.destroy_window()

def visualize_cameras_pyramids(positions):
    """
    Visualize the cameras with pyramids.

    :param R: Rotation matrix
    :param T: Translation vector
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # First camera in the origin
    camera1 = create_camera_pyramid()
    vis.add_geometry(camera1)

    for pos in positions:
        # Second camera with the calculated rotation and translation
        camera = create_camera_pyramid()

        # Combine the rotation and translation into a single transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = T.T

        # Apply the transformation to the pyramid for the second camera
        camera.transform(transform)
        vis.add_geometry(camera)

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
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :], 
            y2 * P2[2, :] - P2[1, :]  
        ])

        # Solve for the 3D point using SVD (last column of Vt gives x)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :]

        # Dehomogenize the 3D point
        X = X[:3] / X[3]
        points_3D.append(X)

    return np.array(points_3D)

def visualize_3D_points(points_3D, colors):
    """
    Visualize the 3D pointcloud with color.

    :param points_3D: Nx3 array of 3D points
    :param colors: Nx3 array of RGB colors
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
    """
    Get the colors of the points in the images.

    :param img1: First image (color)
    :param img2: Second image (color)
    :param pts1: 2D points from camera 1 (Nx2 array)
    :param pts2: 2D points from camera 2 (Nx2 array)
    :return: Nx3 array of colors
    """

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

# Extra mesh
def create_mesh_from_pointcloud(pcd, method='poisson', depth=8, radii=[0.005, 0.01, 0.02]):
    """
    Create a mesh from a point cloud using Poisson or Ball Pivoting.

    :param pcd: Open3D point cloud object
    :param method: 'poisson' or 'ball_pivoting'
    :param depth: Depth for Poisson reconstruction
    :param radii: Radii list for Ball Pivoting
    :return: Open3D mesh object
    """
    print(f"Estimating normals for mesh reconstruction...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    if method == 'poisson':
        print(f"Running Poisson reconstruction with depth={depth}...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        return mesh
    elif method == 'ball_pivoting':
        print(f"Running Ball Pivoting reconstruction...")
        pcd = pcd.voxel_down_sample(voxel_size=0.001)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        return mesh
    else:
        raise ValueError("Method must be 'poisson' or 'ball_pivoting'")
    
def visualize_mesh_with_callback(mesh):
    """
    Visualize the mesh and allow closing with the 'C' key.

    :param mesh: Open3D mesh object
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.register_key_callback(ord("C"), close_visualizer)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":

    triangulate_method = sys.argv[1].lower()

    number_of_cameras = 2

    image_paths = get_image_paths(number_of_cameras)
    loaded_images = read_images(image_paths)
    print(f"Number of cameras: {len(loaded_images)}")
    print(f"Number of images per camera: {len(loaded_images[0])}")

    results = decode_gray_pattern_n_cameras(loaded_images)

    color_images = []
    for i in range(len(results)):
        color_images.append(cv2.resize(cv2.imread(image_paths[i][0], cv2.IMREAD_COLOR), (1920, 1080)))

    correspondences = find_correspondences_n_cameras(results)

    K = np.array([[9.51663140e+03, 0.00000000e+00, 2.81762458e+03],[0.00000000e+00, 8.86527952e+03, 1.14812762e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    essential_matrices = []
    masks  = []
    pts1_list = []
    pts2_list = []
    for c in correspondences:
        keypoints1, keypoints2, matches = c
        E, mask, pts1, pts2 = compute_essential_matrix(keypoints1, keypoints2, matches, K)
        essential_matrices.append(E)
        masks.append(mask)
        pts1_list.append(pts1)
        pts2_list.append(pts2)

    # R = rotatie vector
    # T = translatie vector
    recovered_poses = []
    print('Recovering poses from essential matrices')
    for i, e in enumerate(essential_matrices):
        _, R, T, mask = cv2.recoverPose(e, pts1_list[i], pts2_list[i], K, mask=masks[i])
        recovered_poses.append((R, T))

    visualize_cameras_axises(recovered_poses)
    visualize_cameras_pyramids(recovered_poses)

    print("Triagulating points")
    points_3d = []
    for i in range(len(pts1_list)):
        pts3d = []
        if triangulate_method == "opencv":
            pts3d = triangulate_opencv(pts1_list[i], pts2_list[i], K, recovered_poses[i][0], recovered_poses[i][1])
        elif triangulate_method == "manual":
            pts3d = triangulate_manual(pts1_list[i], pts2_list[i], K, recovered_poses[i][0], recovered_poses[i][1])
        points_3d.append(pts3d)

    
    print("Getting colors")
    colors = []
    for i in range(len(results) - 1):
        for j in range(i + 1, len(results)):
            colors.append(get_colors(color_images[i], color_images[j], pts1_list[i], pts2_list[i]))

    pts     = np.vstack(points_3d)   # (M,3)
    colors  = np.vstack(colors)      # (M,3), corresponderend met pts

    # 1) ronde en unique zoals eerder
    pts_rnd    = np.round(pts, 3)
    structured = np.ascontiguousarray(pts_rnd).view(
        np.dtype((np.void, pts_rnd.dtype.itemsize * 3))
    )
    _, idx = np.unique(structured, return_index=True)

    # 2) pas idx toe op beide arrays
    pts_unique   = pts[idx]
    colors_unique = colors[idx]

    # 3D punten visualiseren
    visualize_3D_points(pts_unique, colors_unique)

    # Extra mesh
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_unique)
    pcd.colors = o3d.utility.Vector3dVector(colors_unique)

    # Generate mesh from point cloud
    mesh = create_mesh_from_pointcloud(pcd, method='poisson', depth=9)

    # Visualize the mesh
    print("Visualizing mesh...")
    visualize_mesh_with_callback(mesh)

