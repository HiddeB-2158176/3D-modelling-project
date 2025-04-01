import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import glob
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

from step_2 import decode_gray_pattern, find_correspondences
from step_3 import compute_essential_matrix, triangulate_opencv, triangulate_manual, get_colors

class PlaneSweeping:
    def __init__(self, cameras, images, virtual_camera, min_depth, max_depth, num_layers=100):
        """
        Initialize the plane sweeping algorithm.
        
        Parameters:
        - cameras: List of camera matrices (projection matrices) for input views
        - images: List of input images
        - virtual_camera: Projection matrix for the virtual camera
        - min_depth: Minimum depth for plane sweeping
        - max_depth: Maximum depth for plane sweeping
        - num_layers: Number of depth layers to use
        """
        self.cameras = cameras
        self.images = images
        self.virtual_camera = virtual_camera
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_layers = num_layers
        
        # Compute image dimensions
        self.height, self.width = images[0].shape[:2]
        
        # Create 2D image plane corners for the virtual camera
        self.image_corners_2d = np.array([
            [0, 0],
            [self.width - 1, 0],
            [self.width - 1, self.height - 1],
            [0, self.height - 1]
        ], dtype=np.float32)
    
    def center_depth_layers(self, point_cloud):
        """
        Center the depth layers around the scene using the point cloud.
        
        Parameters:
        - point_cloud: 3D point cloud of the scene (Nx3 array)
        
        Returns:
        - centered_min_depth: Adjusted minimum depth
        - centered_max_depth: Adjusted maximum depth
        """
        # Calculate the center of the point cloud
        center = np.mean(point_cloud, axis=0)
        
        # Project the center to get its depth in the virtual camera
        center_homogeneous = np.append(center, 1)
        projected_center = np.dot(self.virtual_camera, center_homogeneous)
        center_depth = projected_center[2]
        
        # Adjust depth range to be centered around the scene
        depth_range = self.max_depth - self.min_depth
        centered_min_depth = max(0.1, center_depth - depth_range/2)
        centered_max_depth = center_depth + depth_range/2
        
        return centered_min_depth, centered_max_depth
    
    def compute_depth_plane(self, depth):
        """
        Compute the 3D depth plane for a given depth.
        
        Parameters:
        - depth: The depth value for the plane
        
        Returns:
        - corners_3d: 3D coordinates of the depth plane corners
        """
        # Get the camera intrinsics from the projection matrix
        K = self.virtual_camera[:3, :3]
        R = self.virtual_camera[:3, :3]
        t = self.virtual_camera[:3, 3]
        
        # Inverse of K to deproject
        K_inv = np.linalg.inv(K)
        
        # Deproject the 2D image corners to 3D at the specified depth
        corners_3d = []
        for corner in self.image_corners_2d:
            # Convert to homogeneous coordinates
            corner_homogeneous = np.append(corner, 1)
            
            # Deproject to get a ray direction
            ray = np.dot(K_inv, corner_homogeneous)
            
            # Scale the ray to the desired depth
            # The ray points from the camera center to the 3D point
            ray_normalized = ray / np.linalg.norm(ray)
            point_3d = ray_normalized * depth
            
            corners_3d.append(point_3d)
        
        return np.array(corners_3d)
    
    def project_to_camera(self, points_3d, camera_matrix):
        """
        Project 3D points onto a camera's image plane.
        
        Parameters:
        - points_3d: 3D points to project
        - camera_matrix: Projection matrix of the camera
        
        Returns:
        - points_2d: Projected 2D points
        """
        points_2d = []
        for point in points_3d:
            # Convert to homogeneous coordinates
            point_homogeneous = np.append(point, 1)
            
            # Project to image plane
            projected_point = np.dot(camera_matrix, point_homogeneous)
            
            # Convert to inhomogeneous coordinates
            point_2d = projected_point[:2] / projected_point[2]
            
            points_2d.append(point_2d)
        
        return np.array(points_2d, dtype=np.float32)
    
    def compute_homography(self, corners_3d, camera_matrix):
        """
        Compute the homography between a camera's image plane and a 3D plane.
        
        Parameters:
        - corners_3d: 3D corners of the depth plane
        - camera_matrix: Projection matrix of the camera
        
        Returns:
        - H: Homography matrix
        """
        # Project the 3D corners to the camera image plane
        projected_corners = self.project_to_camera(corners_3d, camera_matrix)
        
        # Find the homography between the original image corners and the projected corners
        H, _ = cv2.findHomography(self.image_corners_2d, projected_corners)
        
        return H
    
    def warp_image(self, image, homography):
        """
        Apply a homography to warp an image.
        
        Parameters:
        - image: The image to warp
        - homography: The homography matrix to apply
        
        Returns:
        - warped_image: The warped image
        """
        # Warp the image using the homography
        warped_image = cv2.warpPerspective(
            image, 
            homography, 
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped_image
    
    def compute_error(self, warped_images):
        """
        Compute the error between warped images for consensus.
        
        Parameters:
        - warped_images: List of warped images for a specific depth
        
        Returns:
        - error_map: Error map showing differences between images
        """
        error_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Calculate pairwise errors and sum them
        for i in range(len(warped_images)):
            for j in range(i+1, len(warped_images)):
                # Calculate absolute difference between the images
                diff = cv2.absdiff(warped_images[i], warped_images[j])
                
                # For color images, sum across channels
                if len(diff.shape) > 2:
                    diff = np.sum(diff, axis=2)
                
                # Add to the error map
                error_map += diff
        
        return error_map
    
    def render(self, point_cloud=None):
        """
        Render a new view using plane sweeping.
        
        Parameters:
        - point_cloud: Optional point cloud for depth layer centering
        
        Returns:
        - novel_view: Rendered novel view
        - depth_map: Corresponding depth map
        """
        # Center depth layers if point cloud is provided
        if point_cloud is not None:
            self.min_depth, self.max_depth = self.center_depth_layers(point_cloud)
        
        # Initialize the result image and depth map
        novel_view = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth_map = np.zeros((self.height, self.width), dtype=np.float32)
        min_error_map = np.full((self.height, self.width), np.inf, dtype=np.float32)
        
        # Compute depth values for each layer
        depths = np.linspace(self.min_depth, self.max_depth, self.num_layers)
        
        # Iterate over depth layers
        for depth in depths:
            # Compute the 3D depth plane for this depth
            corners_3d = self.compute_depth_plane(depth)
            
            # Warp each input image to this depth plane
            warped_images = []
            for i in range(len(self.cameras)):
                # Compute homography for this camera and depth
                H = self.compute_homography(corners_3d, self.cameras[i])
                
                # Warp the image
                warped_image = self.warp_image(self.images[i], H)
                warped_images.append(warped_image)
            
            # Compute error map for this depth
            error_map = self.compute_error(warped_images)
            
            # Update novel view where the error is lower than the current minimum
            update_mask = error_map < min_error_map
            
            for i in range(3):  # For each color channel
                # Take the average of all warped images for pixels where error is minimal
                avg_image = np.mean([img[:,:,i] for img in warped_images], axis=0)
                novel_view[:,:,i][update_mask] = avg_image[update_mask]
            
            # Update the minimum error map and depth map
            depth_map[update_mask] = depth
            min_error_map[update_mask] = error_map[update_mask]
        
        return novel_view, depth_map

if __name__ == "__main__":
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

    img1ColorRGB = cv2.cvtColor(img1Color, cv2.COLOR_BGR2RGB)
    img2ColorRGB = cv2.cvtColor(img2Color, cv2.COLOR_BGR2RGB)

    keypoints1, keypoints2, matches = find_correspondences(result0, result1)

    K = np.array([[9.51663140e+03, 0.00000000e+00, 2.81762458e+03],[0.00000000e+00, 8.86527952e+03, 1.14812762e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    E, mask, pts1, pts2 = compute_essential_matrix(keypoints1, keypoints2, matches, K)

    # R = rotatie vector
    # T = translatie vector
    _, R, T, mask = cv2.recoverPose(E, pts1, pts2, K)

    points_3D_opencv = triangulate_opencv(pts1, pts2, K, R, T)
    #points_3D_manual = triangulate_manual(pts1, pts2, K, R, T)

    colors = get_colors(img1Color, img2Color, pts1, pts2)

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera (identity pose)
    P2 = K @ np.hstack((R, T))  # Second camera (transformed pose)
    
    # Load camera matrices
    cameras = [
        P1,
        P2
    ]


    # Create a virtual camera (e.g., halfway between the two cameras)
    alpha = 0.5  # Change this to move between camera positions (0 = first camera, 1 = second camera)

    # Create rotation objects
    r1 = Rot.from_matrix(np.eye(3))
    r2 = Rot.from_matrix(R)

    # Set up the interpolation
    key_times = [0, 1]
    key_rots = Rot.from_matrix(np.stack([np.eye(3), R]))

    # Create the slerp interpolator
    slerp = Slerp(key_times, key_rots)

    # Get interpolated rotation
    R_interp = slerp([alpha])[0].as_matrix()

    # Interpolate translation
    t_interp = alpha * T

    # Create virtual camera matrix
    P_virtual = np.hstack((R_interp, t_interp))
    P_virtual = K @ P_virtual

    # print camera 1 rotation and translation
    print("Camera 1 rotation matrix:\n", np.eye(3))
    print("Camera 1 translation vector:\n", np.zeros((3, 1)))
    print("Camera 1 projection matrix:\n", P1)

    # print camera 2 rotation and translation
    print("Camera 2 rotation matrix:\n", R)
    print("Camera 2 translation vector:\n", T)
    print("Camera 2 projection matrix:\n", P2)

    # check the rotation and translation of the virtual camera
    print("Virtual camera rotation matrix:\n", R_interp)
    print("Virtual camera translation vector:\n", t_interp)
    print("Virtual camera projection matrix:\n", P_virtual)

    # Check the relative positions of your cameras
    print("Camera 1 position:", P1[ :3, 3])
    print("Camera 2 position:", P2[ :3, 3])
    print("Virtual camera position:", P_virtual[:3, 3])

    images = [img1Color, img2Color]

    
    # Initialize plane sweeping
    ps = PlaneSweeping(
        cameras=cameras,
        images=images,
        virtual_camera=P_virtual,
        min_depth=5,
        max_depth=15,
        num_layers=50
    )
    
    # Render novel view
    novel_view, depth_map = ps.render(points_3D_opencv)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img1Color, cv2.COLOR_BGR2RGB))
    plt.title('Left Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(novel_view, cv2.COLOR_BGR2RGB))
    plt.title('Novel View')
    
    plt.subplot(1, 3, 3)
    plt.imshow(depth_map, cmap='jet')
    plt.title('Depth Map')
    
    plt.tight_layout()
    plt.show()

# The example usag