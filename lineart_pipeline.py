import cv2
import numpy as np

def custom_lineart_pipeline(img):
    # 1. Gaussian blur (50) - kernel must be odd, so use 51
    blur1 = cv2.GaussianBlur(img, (51, 51), 0)
    # 2. Grayscale
    gray = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
    # 3. Adaptive threshold (block size 55, C value 14, Gaussian)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 55, 14
    )
    # 4. Median blur (5)
    median1 = cv2.medianBlur(adaptive, 5)
    # 5. Gaussian blur (22, kernel must be odd, so use 23)
    blur2 = cv2.GaussianBlur(median1, (23, 23), 0)
    # 6. Median blur (5)
    median2 = cv2.medianBlur(blur2, 5)
    return median2
