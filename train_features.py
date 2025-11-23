import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import skew

# Configuration
DATA_DIR = 'capturas'
MODEL_PATH = 'feature_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
CLASS_NAMES = ['botella', 'carton', 'lata']

def extract_features(image):
    """
    Extracts a feature vector from an image.
    Features (Shape & Texture Focused):
    1. Color Stats: Mean Sin/Cos Hue, Mean/Std/Max/Skew Saturation, Value (Calculated on Object Mask)
    2. Shape: Aspect Ratio, Solidity, Extent, Circularity, Rectangularity (Calculated on Object Mask)
    3. Texture: Edge Density
    4. Color Uniformity: Circular Variance of Hue
    """
    features = []
    
    # 0. Low Light Enhancement (Match predict_real.py logic)
    hsv_check = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv_check[:, :, 2])
    
    if avg_brightness < 80: # Threshold for low light
        # Gamma Correction
        gamma = 1.5 
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table)

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
    
    if avg_corner_brightness > 127:
        # Light background -> Object is darker -> Invert for mask
        thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        # Dark background -> Object is lighter
        thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        
    # 2. Segmentation (Get Object Mask)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, thresh_type)
    
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
        
        h_channel = hsv[:, :, 0].astype(float)
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
        circ_var_h = 1
        mean_s = 0
        mean_v = 0
        std_s = 0
        std_v = 0
        max_s = 0
        max_v = 0
        skew_s = 0
        skew_v = 0
        
    features.extend([mean_sin_h, mean_cos_h, circ_var_h, mean_s, mean_v, std_s, std_v, max_s, max_v, skew_s, skew_v])
    
    # 5. Texture Features (Edge Density)
    gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_resized, 100, 200)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask_resized)
    edge_density = np.sum(edges_masked) / (edges.shape[0] * edges.shape[1])
    features.append(edge_density)
        
    return np.array(features)

def train_feature_model():
    print("Extracting features from images...")
    X = []
    y = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found.")
            continue
            
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                feats = extract_features(img)
                X.append(feats)
                y.append(class_idx)
                
    X = np.array(X)
    y = np.array(y)
    
    print(f"Extracted features for {len(X)} images.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Classifier
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced_subsample', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature Importance
    importances = model.feature_importances_
    feature_names = ['Aspect Ratio', 'Solidity', 'Extent', 'Circularity', 'Rectangularity', 
                     'Mean Sin Hue', 'Mean Cos Hue', 'Hue Circ Var', 
                     'Mean Sat', 'Mean Val', 'Std Sat', 'Std Val', 'Max Sat', 'Max Val', 
                     'Skew Sat', 'Skew Val', 'Edge Density']
    
    print("\nFeature Importances:")
    if len(feature_names) == len(importances):
        for name, imp in zip(feature_names, importances):
            print(f"{name}: {imp:.4f}")
    else:
        print(f"Error: Feature names ({len(feature_names)}) and importances ({len(importances)}) count mismatch.")
        print(importances)
    
    # Save model and scaler
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

if __name__ == '__main__':
    train_feature_model()
