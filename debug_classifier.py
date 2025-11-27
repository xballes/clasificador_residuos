#!/usr/bin/env python3
"""
Script de debug mejorado para analizar por qué se detectan objetos falsos
"""

import cv2
import numpy as np
from feature_extractor import FeatureExtractor
from roi_detector import ROIDetector
from object_segmenter import ObjectSegmenter
from waste_classifier import WasteClassifier

# Cargar imagen
image_path = 'capturas_buenas/real/undist_1764002774.png'
image = cv2.imread(image_path)

print(f"Analizando: {image_path}")
print(f"Tamaño: {image.shape}\n")

# Crear módulos
roi_detector = ROIDetector(margin=20)
segmenter = ObjectSegmenter(min_area=1000)
feature_extractor = FeatureExtractor()
classifier = WasteClassifier(confidence_threshold=0.4)

# ROI
roi_mask, roi_info = roi_detector.create_roi_mask(image, detect_aruco=True, detect_box=True)
print(f"ROI: {len(roi_info['aruco_markers'])} ArUco markers, Caja: {roi_info['box_region'] is not None}\n")

# Segmentar
contours, seg_info = segmenter.segment_objects(image, roi_mask=roi_mask, debug=True)
print(f"Objetos detectados: {len(contours)}\n")

# Analizar cada objeto
for i, contour in enumerate(contours):
    print(f"{'='*70}")
    print(f"OBJETO #{i+1}")
    print(f"{'='*70}")
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    
    print(f"\nGEOMETRÍA:")
    print(f"  Bounding Box: x={x}, y={y}, w={w}, h={h}")
    print(f"  Área: {area:.0f}")
    
    # Analizar color del objeto
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    obj_pixels = hsv[mask > 0]
    
    if len(obj_pixels) > 0:
        mean_h = np.mean(obj_pixels[:, 0])
        mean_s = np.mean(obj_pixels[:, 1])
        mean_v = np.mean(obj_pixels[:, 2])
        std_v = np.std(obj_pixels[:, 2])
        
        print(f"\nANÁLISIS DE COLOR (para filtrado de fondo):")
        print(f"  Brillo medio: {mean_v:.1f} (oscuro si < 50)")
        print(f"  Saturación media: {mean_s:.1f} (baja si < 20)")
        print(f"  Variación de brillo: {std_v:.1f} (uniforme si < 15)")
        
        is_very_dark = mean_v < 50
        is_very_uniform = std_v < 15
        is_low_saturation = mean_s < 20
        
        background_score = sum([is_very_dark, is_very_uniform, is_low_saturation])
        
        print(f"\n  ¿Es muy oscuro? {is_very_dark}")
        print(f"  ¿Es muy uniforme? {is_very_uniform}")
        print(f"  ¿Tiene baja saturación? {is_low_saturation}")
        print(f"  Background score: {background_score}/3 (rechazar si >= 2)")
        
        if background_score >= 2:
            print(f"  ⚠️ DEBERÍA SER RECHAZADO COMO FONDO")
        else:
            print(f"  ✓ Pasa el filtro de fondo")
    
    # Extraer características
    features = feature_extractor.extract_features(image, contour)
    
    # Clasificar
    class_name, confidence, scores = classifier.classify(features)

    # --- DEBUG EXTRA DE FEATURES CLAVE ---
    elong = features.get("elongation_ratio", 0.0)
    ar = features.get("aspect_ratio", 0.0)
    circ = features.get("circularity", 0.0)
    spec_top = features.get("specular_ratio_top", 0.0)
    edge_top = features.get("edge_density_top", 0.0)
    edge_body = features.get("edge_density_body", 0.0)
    
    print("\nFEATURES CLAVE:")
    print(f"  Elongation ratio: {elong:.2f}")
    print(f"  Aspect ratio (w/h): {ar:.2f}")
    print(f"  Circularity: {circ:.3f}")
    print(f"  Specular ratio TOP: {spec_top:.4f}")
    print(f"  Edge density TOP: {edge_top:.4f}")
    print(f"  Edge density BODY: {edge_body:.4f}")

    
    print(f"\nCLASIFICACIÓN:")
    print(f"  Clase predicha: {class_name}")
    print(f"  Confianza: {confidence:.2%}")
    print(f"  Scores: LATA={scores['LATA']:.1f}, BOTELLA={scores['BOTELLA']:.1f}, CARTON={scores['CARTON']:.1f}")
    
    # Guardar
    margin = 20
    y1 = max(0, y - margin)
    y2 = min(image.shape[0], y + h + margin)
    x1 = max(0, x - margin)
    x2 = min(image.shape[1], x + w + margin)
    
    obj_roi = image[y1:y2, x1:x2].copy()
    contour_shifted = contour - [x1, y1]
    cv2.drawContours(obj_roi, [contour_shifted], -1, (0, 255, 0), 2)
    cv2.imwrite(f'results/debug_object_{i+1}.png', obj_roi)
    print(f"  Guardado en: results/debug_object_{i+1}.png")
    
    print()

print(f"\n{'='*70}")
print("ANÁLISIS COMPLETO")
print(f"{'='*70}")
