import cv2
import numpy as np

# Load the problematic image
img_path = r'capturas_buenas\real\undist_1764002945.png'
original_img = cv2.imread(img_path)

if original_img is None:
    print(f"Error: Could not load image from {img_path}")
    exit()

print(f"Image shape: {original_img.shape}")

# Check for low light conditions
hsv_check = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
avg_brightness = np.mean(hsv_check[:, :, 2])
print(f"Average brightness: {avg_brightness:.1f}")

if avg_brightness < 80:
    print("Applying gamma correction...")
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    original_img = cv2.LUT(original_img, table)

# Preprocessing for Segmentation
gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# NEW MULTI-STRATEGY SEGMENTATION
print("\n=== Testing New Multi-Strategy Segmentation ===")

# Strategy 1: Color-based segmentation
hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
mask_colored = cv2.inRange(hsv[:, :, 1], 30, 255)
print(f"Color mask non-zero pixels: {cv2.countNonZero(mask_colored)}")

# Strategy 2: Edge-based detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 100)
kernel_edge = np.ones((3, 3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel_edge, iterations=2)
print(f"Edge mask non-zero pixels: {cv2.countNonZero(edges_dilated)}")

# Strategy 3: Adaptive thresholding
adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 5)
print(f"Adaptive mask non-zero pixels: {cv2.countNonZero(adaptive)}")

# Combine all strategies
binary = cv2.bitwise_or(mask_colored, edges_dilated)
binary = cv2.bitwise_or(binary, adaptive)
print(f"Combined mask non-zero pixels: {cv2.countNonZero(binary)}")

# Clean up
kernel_clean = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean, iterations=1)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean, iterations=2)
print(f"After cleanup non-zero pixels: {cv2.countNonZero(binary)}")

# Morphological operations (from original code)
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
print(f"After opening non-zero pixels: {cv2.countNonZero(opening)}")

# Find contours
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nNumber of contours found: {len(contours)}")

MIN_AREA = 1000
valid_contours = []
if contours:
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            print(f"  Contour {i}: area = {area:.0f} ✓ (above MIN_AREA)")
            valid_contours.append(cnt)
        else:
            print(f"  Contour {i}: area = {area:.0f} ✗ (below MIN_AREA)")

print(f"\nValid contours (area > {MIN_AREA}): {len(valid_contours)}")

# Save debug images
cv2.imwrite('debug_new_colored.png', mask_colored)
cv2.imwrite('debug_new_edges.png', edges_dilated)
cv2.imwrite('debug_new_adaptive.png', adaptive)
cv2.imwrite('debug_new_combined.png', binary)
cv2.imwrite('debug_new_final.png', opening)

# Create visualization with contours
vis_img = original_img.copy()
if valid_contours:
    cv2.drawContours(vis_img, valid_contours, -1, (0, 255, 0), 2)
cv2.imwrite('debug_new_with_contours.png', vis_img)

# Create side-by-side comparison
vis = np.hstack([
    cv2.cvtColor(mask_colored, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
])
cv2.imwrite('debug_new_strategies.png', vis)

print("\nDebug images saved:")
print("  - debug_new_colored.png (color-based)")
print("  - debug_new_edges.png (edge-based)")
print("  - debug_new_adaptive.png (adaptive threshold)")
print("  - debug_new_combined.png (combined)")
print("  - debug_new_final.png (after morphology)")
print("  - debug_new_with_contours.png (detected objects)")
print("  - debug_new_strategies.png (comparison)")
