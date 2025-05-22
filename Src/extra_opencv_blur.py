import cv2
import numpy as np
import glob

def box_blur(img, k=1):
    pad = k // 2
    img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = img_padded[i:i+k, j:j+k].mean(axis=(0, 1))
    return out


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
    
    # Load color images for final visualization
    img1Color = cv2.resize(cv2.imread(images_view0[0], cv2.IMREAD_COLOR), (960, 540))
    img2Color = cv2.resize(cv2.imread(images_view1[0], cv2.IMREAD_COLOR), (960, 540))

    # Apply box blur
    blurred_img = box_blur(img1Color, k=10)
    blurred_img_cv2 = cv2.blur(img1Color, (10, 10))

    cv2.imshow('Blurred Image', blurred_img)
    cv2.imshow('Blurred Image (OpenCV)', blurred_img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    diff = cv2.absdiff(blurred_img, blurred_img_cv2)
    cv2.imshow("Difference", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()