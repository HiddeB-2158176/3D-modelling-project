import numpy as np
import cv2 as cv
import glob

DATA_PATH = '../Data/GrayCodes/'

def find_chessboard_corners(images, scale):
    """
    Find the chessboard corners in the images and calibrate the camera.

    :param images: List of image paths.
    :param scale: Scale factor for resizing the images.
    :return: ret, K, dist, rvecs, tvecs, objpoints, imgpoints
    """
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.  

    # Arrays to store object points and image points from all the images.
    for fname in images:
        # Read the image
        img = cv.imread(fname)
        small_img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        gray = cv.cvtColor(small_img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,9), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            corners2 = corners2 / scale
            imgpoints.append(corners2)

    #         # Draw and display the corners
    #         cv.drawChessboardCorners(img, (7,9), corners2, ret)
    #         cv.imshow('img', img)
    #         cv.waitKey(500)
    # cv.destroyAllWindows()

    h,w = img.shape[:2]
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w,h), None, None)
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
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h)) # Get the new camera matrix and region of interest
    mapx, mapy = cv.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5) # Get the map for remapping

    # Undistort the images and save them
    for i, fname in enumerate(images):
        img = cv.imread(fname)
        undistorded = cv.remap(img,mapx,mapy,cv.INTER_LINEAR)
        x,y,w,h = roi
        undistorded = undistorded[y:y+h, x:x+w]
        cv.imwrite("../Result/undistorted/" + str(i) + '.png', undistorded)
        cv.waitKey(500)
    cv.destroyAllWindows()

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

def calculate_pose(K, dist, rvecs, tvecs):
    """
    Calculate the pose of the camera for each image and save the images.

    :param K: Camera matrix.
    :param dist: Distortion coefficients.
    :param rvecs: Rotation vectors.
    :param tvecs: Translation vectors.
    """
    for i in range(len(images)):
        img = cv.imread(images[i])
        img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (7,9), None)

        if ret == True:
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, K, dist)
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, K, dist)
            img = draw(img, corners2, imgpts)
            cv.imwrite('../Result/chess_poses/' + str(i) + '.png', img)

def save_params(rvecs, tvecs, K, dist):
    """
    Save the camera parameters to a text file.

    :param rvecs: Rotation vectors.
    :param tvecs: Translation vectors.
    :param K: Camera matrix.
    :param dist: Distortion coefficients.
    """
    with open('../Result/camera_parameters.txt', 'w') as f:
        f.write(f"Camera Matrix (K):\n{K}\n")
        f.write(f"Distortion Coefficients:\n{dist}\n")

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R, _ = cv.Rodrigues(rvec)  # Convert rotation vector to rotation matrix
            f.write(f"\nPose {i + 1}:\n")
            f.write(f"Rotation Matrix:\n{R}\n")
            f.write(f"Translation Vector:\n{tvec}\n")
                

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)

# Create a 3D axis for each image
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

scale = 0.5
images = glob.glob('../Data/GrayCodes/chess/*.jpg')

# Calibration camera
ret, K, dist, rvecs, tvecs, objpoints, imgpoints = find_chessboard_corners(images, scale)
#undistort_images(images, K, dist)
#calculate_pose(K, dist, rvecs, tvecs)
save_params(rvecs, tvecs, K, dist)


# print("ret = ", ret)
# print("K = ", K) # intrinsic matrix
# print("dist = ", dist) # lens distortion
print("rvecs = ", rvecs) # rotation vectors voor elke img
print("tvecs = ", tvecs) # translation vectors voor elke img