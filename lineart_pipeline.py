import cv2
import numpy as np

def custom_lineart_pipeline(img):
    # 1. Gaussian Blur (intensity: 50)
    blur1 = cv2.GaussianBlur(img, (51, 51), 0)  # must be odd; 50 rounded to 51

    # 2. Grayscale (intensity: 100, just convert)
    gray = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)

    # 3. Adaptive Threshold (block size: 55, c: 14, method: gaussian)
    adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        55, 14
    )

    # 4. Median Blur (kernelSize: 5)
    median1 = cv2.medianBlur(adapt, 5)

    # 5. "Sketch #1" (Canny edge)
    edges = cv2.Canny(median1, 50, 150)

    # 6. Gaussian Blur (intensity: 22)
    blur2 = cv2.GaussianBlur(edges, (23, 23), 0)  # 22 rounded to 23

    # 7. Median Blur (kernelSize: 5)
    median2 = cv2.medianBlur(blur2, 5)

    # INVERT for coloring page look
    out = cv2.bitwise_not(median2)
    return out
