import numpy as np
import cv2 as cv
import glob
import os

def find_chessboard_corners(images, scale):
    """
    Find the chessboard corners in the images and calibrate the camera.

    :param images: List of image paths.
    :param scale: Scale factor for resizing the images.
    :return: ret, K, dist, rvecs, tvecs, objpoints, imgpoints
    """

    objpoints = []
    imgpoints = []

    objp = np.zeros((13*8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:13, 0:8].T.reshape(-1, 2)

    print(f"Processing {len(images)} images...")

    for i, fname in enumerate(images):
        img = cv.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}")
            continue

        if scale != 1.0:
            img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (13, 8), None)

        if ret:
            print(f"[{i}] Chessboard found in {fname}")
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            if scale != 1.0:
                corners2 = corners2 / scale
            objpoints.append(objp)
            imgpoints.append(corners2)
        else:
            print(f"[{i}] Chessboard NOT found in {fname}")

    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise ValueError("No valid chessboard corners found in any image. Check pattern size and image quality.")

    h, w = img.shape[:2]
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    return ret, K, dist, rvecs, tvecs, objpoints, imgpoints


def undistort_images(images, K, dist):
    """
    Undistort the images using the camera matrix and distortion coefficients.

    :param images: List of image paths.
    :param K: Camera matrix.
    :param dist: Distortion coefficients.
    """
    sample = cv.imread(images[0])
    h, w = sample.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    mapx, mapy = cv.initUndistortRectifyMap(K, dist, None, newcameramtx, (w, h), 5)

    os.makedirs("../Result/undistortedNew", exist_ok=True)

    for i, fname in enumerate(images):
        img = cv.imread(fname)
        undistorted = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
        cv.imwrite(f"../Result/undistortedNew/{i}.png", undistorted)

def draw(img, corners, imgpts):
    """
    Draw the 3D axis on the image.

    :param img: Image to draw on.
    :param corners: Corners of the chessboard.
    :param imgpts: Image points.
    :return: Image with 3D axis drawn on it.
    """
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def calculate_pose(K, dist, images, scale):
    """
    Calculate the pose of the camera for each image and save the images.

    :param K: Camera matrix.
    :param dist: Distortion coefficients.
    :param images: List of image paths.
    :param scale: Scale factor for resizing the images.
    """
    objp = np.zeros((13*8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:14, 0:9].T.reshape(-1, 2)

    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1, 3)

    os.makedirs("../Result/chess_poses_new", exist_ok=True)

    for i in range(len(images)):
        img = cv.imread(images[i])
        if scale != 1.0:
            img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (13, 8), None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            if scale != 1.0:
                corners2 = corners2 / scale
            ret, rvec, tvec = cv.solvePnP(objp, corners2, K, dist)
            imgpts, _ = cv.projectPoints(axis, rvec, tvec, K, dist)
            img = draw(img, corners2, imgpts)
            cv.imwrite(f'../Result/chess_poses_new/{i}.png', img)

def save_params(rvecs, tvecs, K, dist):
    """
    Save the camera parameters to a text file.

    :param rvecs: Rotation vectors.
    :param tvecs: Translation vectors.
    :param K: Camera matrix.
    :param dist: Distortion coefficients.
    """
    os.makedirs("../Result", exist_ok=True)
    with open('../Result/camera_parameters_new.txt', 'w') as f:
        f.write(f"Camera Matrix (K):\n{K}\n")
        f.write(f"Distortion Coefficients:\n{dist}\n")

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R, _ = cv.Rodrigues(rvec)
            f.write(f"\nPose {i + 1}:\n")
            f.write(f"Rotation Matrix:\n{R}\n")
            f.write(f"Translation Vector:\n{tvec}\n")

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

scale = 1.0

images = glob.glob('../Result/ownChessImages/*.png')
print(f"Found {len(images)} images.")

ret, K, dist, rvecs, tvecs, objpoints, imgpoints = find_chessboard_corners(images, scale)

save_params(rvecs, tvecs, K, dist)

print("Calibration successful.")
print("K =", K)
print("dist =", dist)
print("rvecs =", rvecs)
print("tvecs =", tvecs)
