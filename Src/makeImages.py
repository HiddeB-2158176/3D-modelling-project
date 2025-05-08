import cv2
import numpy as np
import glob

# Get patterns from the patterns folder
patterns = glob.glob('../patterns/*.jpg')
for i in range(len(patterns)):
    patterns[i] = cv2.imread(patterns[i], cv2.IMREAD_GRAYSCALE)
    patterns[i] = cv2.resize(patterns[i], (1280, 800), interpolation=cv2.INTER_CUBIC)



def show_pattern_fullscreen(pattern_img):
    cv2.namedWindow("Pattern", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Pattern", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Pattern", pattern_img)
    cv2.waitKey(500)  # Delay to allow camera to expose and settle

# from: https://www.kurokesu.com/main/2020/05/22/uvc-camera-exposure-timing-in-opencv/
vc = cv2.VideoCapture(1) # Index may be different
vc.set(cv2.CAP_PROP_FPS, 30.0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
vc.set(cv2.CAP_PROP_EXPOSURE, -6)
vc.set(cv2.CAP_PROP_TEMPERATURE, 3200) # doesn't work in Windows. Not tested in Linux/OSX.
vc.set(cv2.CAP_PROP_AUTO_WB, 0) # doesn't work in Windows. Not tested in Linux/OSX.
# For whitebalance use settings in OBS 

for i, pattern in enumerate(patterns):
    show_pattern_fullscreen(pattern)
    ret, frame = vc.read()
    if ret:
        cv2.imwrite(f"capture_{i:02}.png", frame)

