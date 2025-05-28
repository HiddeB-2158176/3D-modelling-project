import numpy as np
import open3d as o3d

# Create a wireframe representation of a camera
def create_camera_wireframe(scale=0.1):
    points = np.array([
        [0, 0, 0],              # Camera origin
        [1, 1, 2],              # Top-right
        [-1, 1, 2],             # Top-left
        [-1, -1, 2],            # Bottom-left
        [1, -1, 2]              # Bottom-right
    ]) * scale

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from origin to corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # Connecting the square
    ]

    colors = [[1, 0, 0] for _ in range(len(lines))]  # Red for camera lines

    camera = o3d.geometry.LineSet()
    camera.points = o3d.utility.Vector3dVector(points)
    camera.lines = o3d.utility.Vector2iVector(lines)
    camera.colors = o3d.utility.Vector3dVector(colors)

    return camera

def close_visualizer(vis):
    """
    Close the Open3D visualizer.
    """
    vis.close()  # This will close the Open3D window
    return False  # Returning False ensures it stops updating

# Read camera parameters from file
def read_camera_parameters(filepath):
    """
    Read camera parameters from a file.
    """
    poses = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Rotation Matrix" in line:
                R = []
                for j in range(3):
                    # Strip brackets and split numbers
                    clean_line = lines[i + j + 1].strip().replace('[', '').replace(']', '')
                    R.append([float(num) for num in clean_line.split()])

                R = np.array(R)

            if "Translation Vector" in line:
                # Translation vector is vertical — read the next three lines
                t = []
                for j in range(3):
                    clean_line = lines[i + j + 1].strip().replace('[', '').replace(']', '')
                    t.append(float(clean_line))

                t = np.array(t).reshape(3, 1)
                poses.append((R, t))
        
        print(poses)
        return poses
    
# Visualize the cameras
def visualise_cameras():
    poses = read_camera_parameters('..\Result\camera_parameters.txt')

    # Visualize the cameras
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    for R, t in poses:
        camera = create_camera_wireframe()
        
        # Combine R and t into a 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t.flatten()

        camera.transform(transform)
        vis.add_geometry(camera)

    vis.register_key_callback(ord("C"), close_visualizer)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualise_cameras()