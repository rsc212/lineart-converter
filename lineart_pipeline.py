import cv2
import numpy as np

def custom_lineart_pipeline(img):
    # 1. Gaussian blur (50)
    blur1 = cv2.GaussianBlur(img, (51, 51), 0)  # Kernel size must be odd
    # 2. Grayscale
    gray = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
    # 3. Adaptive threshold (block size 55, C value 14, method Gaussian, blend 100)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 55, 14
    )
    # 4. Median blur (5)
    median1 = cv2.medianBlur(adaptive, 5)
    # 5. (sketch #1) Gaussian blur (22)
    blur2 = cv2.GaussianBlur(median1, (23, 23), 0)  # Kernel size must be odd
    # 6. Median blur (5)
    median2 = cv2.medianBlur(blur2, 5)
    return median2
