import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def deproject_corners(K, R, t, image_shape, depth):
    """Deproject corners of image to 3D at given depth"""
    h, w = image_shape
    corners_2d = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    corners_3d = []

    K_inv = np.linalg.inv(K)
    for uv in corners_2d:
        uv_h = np.array([uv[0], uv[1], 1.0])
        ray = K_inv @ uv_h
        ray = ray / np.linalg.norm(ray)
        point_cam = ray * depth
        point_world = R.T @ (point_cam - t)
        corners_3d.append(point_world)

    return np.array(corners_3d)

def project_points(K, R, t, points_3d):
    """Project 3D points onto image plane"""
    proj = K @ (R @ points_3d.T + t.reshape(3, 1))
    proj /= proj[2]
    return proj[:2].T

def compute_plane_sweep(images, Ks, Rs, ts, K_virt, R_virt, t_virt, depth_range, step):
    """Compute plane sweep from multiple images"""
    h, w, _ = images[0].shape
    virtual_image = np.zeros((h, w, 3), dtype=np.uint8)
    min_error = np.full((h, w), np.inf)

    for depth in np.arange(depth_range[0], depth_range[1], step):
        plane_corners_3d = deproject_corners(K_virt, R_virt, t_virt, (h, w), depth)
        for i, image in enumerate(images):
            # Project 3D plane corners to this camera view
            corners_2d = project_points(Ks[i], Rs[i], ts[i], plane_corners_3d)
            src_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            dst_corners = corners_2d.astype(np.float32)

            H, _ = cv2.findHomography(src_corners, dst_corners)
            warped = cv2.warpPerspective(image, H, (w, h))

            if i == 0:
                reference = warped
            else:
                diff = cv2.absdiff(reference, warped)
                error = np.linalg.norm(diff, axis=2)

                update_mask = error < min_error
                min_error[update_mask] = error[update_mask]
                virtual_image[update_mask] = reference[update_mask]

    return virtual_image



img1 = cv2.imread("../Data/GrayCodes/view0/00.jpg")
img2 = cv2.imread("../Data/GrayCodes/view1/00.jpg")

K = [[9.51663140e+03, 0.00000000e+00, 2.81762458e+03],[0.00000000e+00, 8.86527952e+03, 1.14812762e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

images = [img1, img2]            # bijvoorbeeld 2 stereo views
Ks = [K, K]                    # intrinsieke matrices
Rs = [np, R2]                    # rotaties
ts = [t1, t2]                    # translatievectoren

K_virt = ...                     # virtuele camera-instellingen
R_virt = ...                     # bv. interpolatie tussen R1 en R2
t_virt = ...                     # idem

output = compute_plane_sweep(
    images, Ks, Rs, ts,
    K_virt, R_virt, t_virt,
    depth_range=(0.0, 10.0),      # pas dit aan op basis van je scène
    step=0.1
)

cv2.imshow("Plane Sweep", output)
cv2.waitKey(0)
