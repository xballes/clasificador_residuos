import cv2
import numpy as np
import sys

def test_roi_detection(image_path):
    print(f"Testing ROI detection on {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return

    # Strategy 1: Simple Thresholding on V channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    
    # Otsu thresholding
    # Invert v_channel because we want the dark area (black mat) to be white in the mask?
    # Otsu finds the threshold to separate two classes.
    # Dark mat will have low V, bright table will have high V.
    # We want mask of dark area.
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(v_channel, (5, 5), 0)
    
    # Otsu
    thresh_val, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(f"Otsu threshold value: {thresh_val}")
    
    # Morphological cleanup
    kernel = np.ones((7,7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find largest contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vis = image.copy()
    
    if contours:
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_cnt = contours[0]
        
        # Approximate polygon
        epsilon = 0.01 * cv2.arcLength(largest_cnt, True)
        approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
        
        # Draw
        cv2.drawContours(vis, [approx], -1, (0, 255, 0), 3)
        
        # Draw all contours in red to see noise
        cv2.drawContours(vis, contours[1:], -1, (0, 0, 255), 1)
        
    cv2.imwrite("debug_roi_test.png", vis)
    cv2.imwrite("debug_roi_mask.png", binary)
    print("Saved debug_roi_test.png and debug_roi_mask.png")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_roi_detection(sys.argv[1])
    else:
        print("Usage: python debug_roi.py <image_path>")
