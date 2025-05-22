import cv2
import numpy as np
import glob

def sobel(img):
    # Sobel operator in x direction
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    # Sobel operator in y direction
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    # Apply convolution
    grad_x = cv2.filter2D(img, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(img, cv2.CV_64F, sobel_y)


    sobel_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return sobel_magnitude


def normalize_for_display(img):
    img = np.absolute(img)
    img = img / np.max(img) * 255
    return img.astype(np.uint8)


if __name__ == "__main__":
    images_view0 = glob.glob('../Data/GrayCodes/view0/*.jpg')
    images_view1 = glob.glob('../Data/GrayCodes/view1/*.jpg')

    images0 = []
    images1 = []
    for i in range(len(images_view0)):
        resized0 = cv2.resize(cv2.imread(images_view0[i], cv2.IMREAD_GRAYSCALE), (960, 540))
        resized1 = cv2.resize(cv2.imread(images_view1[i], cv2.IMREAD_GRAYSCALE), (960, 540))
        images0.append(resized0)
        images1.append(resized1)
    
    img1Color = cv2.resize(cv2.imread(images_view0[0], cv2.IMREAD_COLOR), (960, 540))
    img2Color = cv2.resize(cv2.imread(images_view1[0], cv2.IMREAD_COLOR), (960, 540))

    img1Gray = cv2.cvtColor(img1Color, cv2.COLOR_BGR2GRAY)
    img2Gray = cv2.cvtColor(img2Color, cv2.COLOR_BGR2GRAY)

    sobel_img = sobel(img1Gray)
    sobel_x = cv2.Sobel(img1Gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img1Gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_img_cv2 = np.sqrt(sobel_x**2 + sobel_y**2)

    # cv2.imshow('Sobel (custom)', normalize_for_display(sobel_img))
    # cv2.imshow('Sobel (OpenCV)', normalize_for_display(sobel_img_cv2))
    cv2.imshow('Sobel (custom)', sobel_img)
    cv2.imshow('Sobel (OpenCV)', sobel_img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    diff = cv2.absdiff(sobel_img.astype(np.uint8), sobel_img_cv2.astype(np.uint8))
    cv2.imshow("Difference", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Apply Sobel operator
    # sobel_img = sobel(img1Gray)
    # sobel_img_cv2 = cv2.Sobel(img1Gray, cv2.CV_64F, 1, 1, ksize=3)

    # cv2.imshow('Sobel Image', sobel_img)
    # cv2.imshow('Sobel Image (OpenCV)', sobel_img_cv2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # diff = cv2.absdiff(sobel_img.astype(np.uint8), sobel_img_cv2.astype(np.uint8))
    # cv2.imshow("Difference", diff)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()