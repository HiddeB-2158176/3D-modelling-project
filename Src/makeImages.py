import cv2
import numpy as np
import glob
import os
from screeninfo import get_monitors
import time

monitors = get_monitors()
if len(monitors) < 2:
    raise RuntimeError("Projector (second screen) not detected. Please extend your display.")
projector_monitor = monitors[1]  # Assuming the projector is the second monitor

# Load Graycode patterns
pattern_files = sorted(glob.glob('../patterns/*.jpg'))
patterns = [cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (1280, 800)) for p in pattern_files]

# add white image and black image to the patterns
patterns.append(np.ones((800, 1280), dtype=np.uint8) * 255)  # White image
patterns.append(np.zeros((800, 1280), dtype=np.uint8))  # Black image

vc = cv2.VideoCapture(0)  # Adjust index as needed
vc.set(cv2.CAP_PROP_FPS, 30.0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
vc.set(cv2.CAP_PROP_EXPOSURE, -8)


def show_pattern_fullscreen(pattern_img):
    cv2.namedWindow("Pattern", cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow("Pattern", projector_monitor.x, projector_monitor.y)
    cv2.setWindowProperty("Pattern", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Pattern", pattern_img)

for i, pattern in enumerate(patterns):
    print(f"Projecting pattern {i+1}/{len(patterns)}")

    show_pattern_fullscreen(pattern)
    cv2.waitKey(200)  # Wait for projector to switch
    time.sleep(0.3)   # Let exposure settle

    ret, frame = vc.read()
    if ret:
        save_path = f"../Result/ownDataset1/view2/capture_{i:02}.png"
        cv2.imwrite(save_path, frame)
        print(f"Captured {save_path}")
    else:
        print(f"Warning: Failed to capture image for pattern {i}")

# capture_count = 35
# for i in range(capture_count):
#     input(f"[{i+1}/{capture_count}] Press Enter to capture...")

#     ret, frame = vc.read()
#     if ret:
#         save_path = f"../Result/ownChessImages/chess_capture_{i:02}.png"
#         cv2.imwrite(save_path, frame)
#         print(f"✅ Saved: {save_path}")
#     else:
#         print("⚠️ Capture failed! Check camera connection or settings.")

cv2.destroyAllWindows()
vc.release()
