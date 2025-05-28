import cv2
import numpy as np
import glob


def to_grayscale(img):
    """
    Convert an image to grayscale using the formula:
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    
    :param img: Input image in BGR format.
    :return: Grayscale image.
    """

    return 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]


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
    
    img1Color = cv2.resize(cv2.imread(images_view0[0], cv2.IMREAD_COLOR), (1920, 1080))
    img2Color = cv2.resize(cv2.imread(images_view1[0], cv2.IMREAD_COLOR), (1920, 1080))

    gray_img_cv2 = cv2.cvtColor(img1Color, cv2.COLOR_BGR2GRAY)
    gray_img = to_grayscale(img1Color).astype(np.uint8)
    cv2.imshow('Grayscale Image', gray_img)
    cv2.imshow('Grayscale Image (OpenCV)', gray_img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    diff = cv2.absdiff(gray_img, gray_img_cv2)
    cv2.imshow("Difference", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()