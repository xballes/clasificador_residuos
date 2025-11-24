import os
import cv2
import numpy as np
import pickle
from scipy.stats import skew

# Configuration
MODEL_PATH = 'feature_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
CLASS_NAMES = ['botella', 'carton', 'lata']
MIN_AREA = 1000

def extract_features(image):
    """
    Extracts a feature vector from an image.
    Features (Shape & Texture Focused):
    1. Color Stats: Mean Sin/Cos Hue, Hue Circular Variance,
       Mean/Std/Max/Skew Saturation, Value (Calculated on Object Mask)
    2. Shape: Aspect Ratio, Solidity, Extent, Circularity, Rectangularity (Calculated on Object Mask)
    3. Texture: Edge Density
    """
    features = []
    
    # 1. Preprocessing & Background Detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check corners to determine background color
    h, w = gray.shape
    corners = [
        gray[0:10, 0:10],
        gray[0:10, w-10:w],
        gray[h-10:h, 0:10],
        gray[h-10:h, w-10:w]
    ]
    avg_corner_brightness = np.mean([np.mean(c) for c in corners])
    
    # Determine threshold type based on background
    if avg_corner_brightness > 127:
        # Light background -> Object is darker -> Invert for mask
        thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        # Dark background -> Object is lighter
        thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        
    # 2. Segmentation (Get Object Mask)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, thresh_type)
    
    # Clean up mask with morphology
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. Shape Features (On Mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape_features = [0, 0, 0, 0, 0]  # Default
    
    if contours:
        # Find largest contour
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if area > 0:
            # Bounding Rect (Axis Aligned)
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            aspect_ratio = float(w_rect) / h_rect
            
            # Convex Hull & Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Extent (Area / BoundingRectArea)
            rect_area = w_rect * h_rect
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            # Circularity
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Rectangularity (Area / MinAreaRect)
            min_rect = cv2.minAreaRect(cnt)
            min_rect_area = min_rect[1][0] * min_rect[1][1]
            rectangularity = float(area) / min_rect_area if min_rect_area > 0 else 0
            
            shape_features = [aspect_ratio, solidity, extent, circularity, rectangularity]
            
    features.extend(shape_features)

    # 4. Color Features (On Mask)
    img_resized = cv2.resize(image, (160, 160))
    mask_resized = cv2.resize(mask, (160, 160), interpolation=cv2.INTER_NEAREST)
    
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    if cv2.countNonZero(mask_resized) > 0:
        mask_bool = mask_resized > 0
        
        # Hue handling: Convert to radians and take sin/cos
        h_channel = hsv[:, :, 0].astype(float)
        # OpenCV Hue is 0-179. Convert to 0-2pi
        h_rad = h_channel * (2 * np.pi / 180.0)
        
        sin_h = np.sin(h_rad)
        cos_h = np.cos(h_rad)
        
        mean_sin_h = np.mean(sin_h[mask_bool])
        mean_cos_h = np.mean(cos_h[mask_bool])

        # Circular Variance = 1 - R (Mean Resultant Length)
        R = np.sqrt(mean_sin_h**2 + mean_cos_h**2)
        circ_var_h = 1 - R
        
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        mean_s = np.mean(s_channel[mask_bool])
        mean_v = np.mean(v_channel[mask_bool])
        
        std_s = np.std(s_channel[mask_bool])
        std_v = np.std(v_channel[mask_bool])
        
        max_s = np.max(s_channel[mask_bool])
        max_v = np.max(v_channel[mask_bool])
        
        skew_s = skew(s_channel[mask_bool])
        skew_v = skew(v_channel[mask_bool])
    else:
        mean_sin_h = 0
        mean_cos_h = 0
        circ_var_h = 1  # Máxima varianza si no hay datos
        mean_s = 0
        mean_v = 0
        std_s = 0
        std_v = 0
        max_s = 0
        max_v = 0
        skew_s = 0
        skew_v = 0
        
    features.extend([
        mean_sin_h,   # 6
        mean_cos_h,   # 7
        circ_var_h,   # 8
        mean_s,       # 9
        mean_v,       # 10
        std_s,        # 11
        std_v,        # 12
        max_s,        # 13
        max_v,        # 14
        skew_s,       # 15
        skew_v        # 16
    ])
    
    # 5. Texture Features (Edge Density)
    gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_resized, 100, 200)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask_resized)
    edge_density = np.sum(edges_masked) / (edges.shape[0] * edges.shape[1])
    features.append(edge_density)
        
    return np.array(features)

    """
    Extracts a feature vector from an image.
    Features (Shape & Texture Focused):
    1. Color Stats: Mean Sin/Cos Hue, Mean/Std/Max/Skew Saturation, Value (Calculated on Object Mask)
    2. Shape: Aspect Ratio, Solidity, Extent, Circularity, Rectangularity (Calculated on Object Mask)
    3. Texture: Edge Density
    """
    features = []
    
    # 1. Preprocessing & Background Detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check corners to determine background color
    h, w = gray.shape
    corners = [
        gray[0:10, 0:10],
        gray[0:10, w-10:w],
        gray[h-10:h, 0:10],
        gray[h-10:h, w-10:w]
    ]
    avg_corner_brightness = np.mean([np.mean(c) for c in corners])
    
    # Determine threshold type based on background
    if avg_corner_brightness > 127:
        # Light background -> Object is darker -> Invert for mask
        thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        # Dark background -> Object is lighter
        thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        
    # 2. Segmentation (Get Object Mask)
    # Use GaussianBlur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, thresh_type)
    
    # Clean up mask with morphology
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. Shape Features (On Mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape_features = [0, 0, 0, 0, 0] # Default
    
    if contours:
        # Find largest contour
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if area > 0:
            # Bounding Rect (Axis Aligned)
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            aspect_ratio = float(w_rect) / h_rect
            
            # Convex Hull & Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Extent (Area / BoundingRectArea)
            rect_area = w_rect * h_rect
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            # Circularity
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Rectangularity (Area / MinAreaRect)
            min_rect = cv2.minAreaRect(cnt)
            min_rect_area = min_rect[1][0] * min_rect[1][1]
            rectangularity = float(area) / min_rect_area if min_rect_area > 0 else 0
            
            shape_features = [aspect_ratio, solidity, extent, circularity, rectangularity]
            
    features.extend(shape_features)

    # 4. Color Features (On Mask)
    img_resized = cv2.resize(image, (160, 160))
    mask_resized = cv2.resize(mask, (160, 160), interpolation=cv2.INTER_NEAREST)
    
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Calculate mean/std/max/skew only on mask pixels
    if cv2.countNonZero(mask_resized) > 0:
        mask_bool = mask_resized > 0
        
        # Hue handling: Convert to radians and take sin/cos
        h_channel = hsv[:, :, 0].astype(float)
        # OpenCV Hue is 0-179. Convert to 0-2pi
        h_rad = h_channel * (2 * np.pi / 180.0)
        
        sin_h = np.sin(h_rad)
        cos_h = np.cos(h_rad)
        
        mean_sin_h = np.mean(sin_h[mask_bool])
        mean_cos_h = np.mean(cos_h[mask_bool])
        
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        mean_s = np.mean(s_channel[mask_bool])
        mean_v = np.mean(v_channel[mask_bool])
        
        std_s = np.std(s_channel[mask_bool])
        std_v = np.std(v_channel[mask_bool])
        
        max_s = np.max(s_channel[mask_bool])
        max_v = np.max(v_channel[mask_bool])
        
        skew_s = skew(s_channel[mask_bool])
        skew_v = skew(v_channel[mask_bool])
    else:
        mean_sin_h = 0
        mean_cos_h = 0
        mean_s = 0
        mean_v = 0
        std_s = 0
        std_v = 0
        max_s = 0
        max_v = 0
        skew_s = 0
        skew_v = 0
        
    features.extend([mean_sin_h, mean_cos_h, mean_s, mean_v, std_s, std_v, max_s, max_v, skew_s, skew_v])
    
    # 5. Texture Features (Edge Density)
    gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_resized, 100, 200)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask_resized)
    edge_density = np.sum(edges_masked) / (edges.shape[0] * edges.shape[1])
    features.append(edge_density)
        
    return np.array(features)

def predict_with_features():
    # Load model and scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or scaler not found. Please run train_features.py first.")
        return

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    print("Model and scaler loaded.")
    
    # Process images in 'capturas/real'
    real_dir = os.path.join('capturas', 'real')
    output_dir = 'output_real'
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(real_dir):
        print(f"Directory {real_dir} not found.")
        return
        
    for filename in os.listdir(real_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(real_dir, filename)
            original_img = cv2.imread(img_path)
            if original_img is None:
                continue

            # Check for low light conditions
            hsv_check = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
            avg_brightness = np.mean(hsv_check[:, :, 2])
            
            if avg_brightness < 80: # Threshold for low light (increased to 80 to be safer)
                print(f"  Low light detected (Brightness: {avg_brightness:.1f}). Enhancing image...")
                # Gamma Correction
                gamma = 1.5 # Values > 1 brighten the image
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                original_img = cv2.LUT(original_img, table)

            
            # Copy for drawing
            draw_img = original_img.copy()
            
            # Preprocessing for Segmentation
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            
            # Check corners to determine background color (Adaptive Thresholding)
            h, w = gray.shape
            corners = [
                gray[0:10, 0:10],
                gray[0:10, w-10:w],
                gray[h-10:h, 0:10],
                gray[h-10:h, w-10:w]
            ]
            avg_corner_brightness = np.mean([np.mean(c) for c in corners])
            
            # Enforce Black Background
            '''if avg_corner_brightness > 80: # Threshold for "Black" background
                print(f"Skipping {filename}: Background too bright ({avg_corner_brightness:.1f})")
                continue'''

            # Dark background -> Object is lighter
            thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, thresh_type)
            
            # Morphological operations to clean up
            kernel = np.ones((5,5), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Watershed Segmentation to separate touching objects
            # sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1
            
            # Now, mark the region of unknown with zero
            markers[unknown == 255] = 0
            
            # Apply Watershed
            markers = cv2.watershed(original_img, markers)
            
            obj_count = 0
            unique_labels = np.unique(markers)
            
            for label in unique_labels:
                if label <= 1: # 0 is boundary, 1 is background
                    continue
                
                # Create a mask for this object
                mask = np.zeros(gray.shape, dtype=np.uint8)
                mask[markers == label] = 255
                
                # Find contours for this mask to get bounding box and shape
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                    
                # There should be only one main contour for this label
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                
                if area > MIN_AREA:
                    x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                    
                    # Border Filter: Ignore objects touching the edges
                    h_img, w_img = original_img.shape[:2]
                    border_margin = 2 # pixels
                    if (x <= border_margin or y <= border_margin or 
                        x + w_rect >= w_img - border_margin or 
                        y + h_rect >= h_img - border_margin):
                        # print(f"  Skipping object at border: {x},{y} {w_rect}x{h_rect}")
                        continue
                    
                    # Create a masked image where everything else is black
                    masked_img = cv2.bitwise_and(original_img, original_img, mask=mask)
                    
                    # Crop to bounding box
                    roi = masked_img[y:y+h_rect, x:x+w_rect]
                    
                    # Extract features
                    feats = extract_features(roi)
                    
                    # Scale features
                    feats_scaled = scaler.transform([feats])
                    
                    # Predict
                    probs = model.predict_proba(feats_scaled)[0]
                    pred_idx = np.argmax(probs)
                    pred_class = CLASS_NAMES[pred_idx]
                    confidence = probs[pred_idx] * 100
                    
                    obj_count += 1
                    print(f"  {filename} (Obj {obj_count}): Found {pred_class} ({confidence:.1f}%)")
                    
                    # Draw contour and label
                    if pred_class == 'botella':
                        color = (100, 0, 0) # Dark Blue
                    elif pred_class == 'carton':
                        color = (19, 69, 139) # Brown
                    elif pred_class == 'lata':
                        color = (128, 128, 128) # Grey
                    else:
                        color = (0, 255, 0) # Default Green

                    # Draw the contour instead of the rectangle
                    cv2.drawContours(draw_img, [cnt], -1, color, 2)
                    
                    label = f"{pred_class} {confidence:.0f}%"
                    # Place text above the bounding box top-left corner
                    cv2.putText(draw_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # --- Specific Detection for Cans ---
                    if pred_class == 'lata':
                        detect_can_features(roi, draw_img, x, y)
            
            # Save result
            output_path = os.path.join(output_dir, f"pred_{filename}")
            cv2.imwrite(output_path, draw_img)
            print(f"Saved prediction to {output_path}")

def detect_can_features(roi_img, draw_img, offset_x, offset_y):
    """
    Detects holes (agujeros) and rings (anillas) in a can ROI.
    Draws directly on draw_img using offsets.
    """
    # Convert to gray for feature detection
    if len(roi_img.shape) == 3:
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi_img

    # --- Hole Detection (Dark circular regions) ---
    # Invert because holes are usually dark
    # Adaptive threshold to find dark spots locally
    thresh_hole = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
    
    # Filter by size and circularity
    contours_hole, _ = cv2.findContours(thresh_hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_hole:
        area = cv2.contourArea(cnt)
        if 50 < area < 1500: # Typical size for a drinking hole
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.6: # Reasonably circular
                    # Found a potential hole
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Draw on main image
                    cv2.rectangle(draw_img, (offset_x + x, offset_y + y), 
                                  (offset_x + x + w, offset_y + y + h), (0, 255, 255), 2)
                    cv2.putText(draw_img, "Agujero", (offset_x + x, offset_y + y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # --- Ring Detection (Pull-tab) ---
    # Rings are often shiny/metallic, so we look for edges or specific shapes
    edges = cv2.Canny(gray_roi, 50, 150)
    contours_ring, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours_ring):
            # Look for contours with children (nested holes)
            if hierarchy[i][2] != -1: 
                area = cv2.contourArea(cnt)
                if 100 < area < 2000: # Typical size for a pull-tab
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h
                    # Pull tabs are often slightly elongated or oval
                    if 0.5 < aspect_ratio < 2.0:
                        # Check solidity - rings have low solidity because of the hole
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            solidity = float(area) / hull_area
                            if solidity < 0.8: # Likely has a hole
                                cv2.rectangle(draw_img, (offset_x + x, offset_y + y), 
                                              (offset_x + x + w, offset_y + y + h), (255, 0, 255), 2)
                                cv2.putText(draw_img, "Anilla", (offset_x + x, offset_y + y - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

if __name__ == '__main__':
    predict_with_features()
