#!/usr/bin/env python3
"""
Script de debug para analizar por qué se filtran objetos en imágenes brillantes
"""

import cv2
import numpy as np
from feature_extractor import FeatureExtractor
from roi_detector import ROIDetector
from object_segmenter import ObjectSegmenter
from waste_classifier import WasteClassifier

# Cargar imagen problemática
image_path = 'capturas_buenas/real/undist_1764003192.png'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: No se pudo cargar la imagen {image_path}")
    exit(1)

print(f"Analizando: {image_path}")
print(f"Tamaño: {image.shape}\n")

# Crear módulos
roi_detector = ROIDetector(margin=20)
segmenter = ObjectSegmenter(min_area=1000)

# ROI
roi_mask, roi_info = roi_detector.create_roi_mask(image, detect_aruco=True, detect_box=True)
print(f"ROI: {len(roi_info['aruco_markers'])} ArUco markers, Caja: {roi_info['box_region'] is not None}\n")

# Segmentar con debug interno
# Vamos a "monkey patch" o simplemente copiar la lógica de filtrado aquí para ver qué pasa
# Primero obtenemos los contornos SIN filtrar (accediendo a métodos internos o modificando el uso)
# Como no puedo modificar la clase fácilmente en runtime para imprimir, voy a replicar la lógica de filtrado aquí.

# 1. Obtener máscara y contornos crudos
mask = segmenter._color_segmentation(image)
if roi_mask is not None:
    mask = cv2.bitwise_and(mask, roi_mask)

# Limpieza
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contornos crudos encontrados: {len(contours)}")

h, w = image.shape[:2]
max_area = (h * w) * 0.5

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    
    print(f"\n--- Contorno #{i+1} ---")
    print(f"  Área: {area:.0f}")
    
    if area < segmenter.min_area:
        print("  RECHAZADO: Área muy pequeña")
        continue
    if area > max_area:
        print("  RECHAZADO: Área muy grande")
        continue
        
    touches_border = (x <= 2 or y <= 2 or x + w_box >= w - 2 or y + h_box >= h - 2)
    if touches_border:
        print("  RECHAZADO: Toca bordes")
        continue

    # Análisis de fondo (la lógica actual)
    mask_cnt = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    obj_pixels = hsv[mask_cnt > 0]
    
    if len(obj_pixels) > 0:
        mean_value = np.mean(obj_pixels[:, 2])
        mean_saturation = np.mean(obj_pixels[:, 1])
        std_value = np.std(obj_pixels[:, 2])
        std_saturation = np.std(obj_pixels[:, 1])
        
        print(f"  Brillo medio: {mean_value:.1f}")
        print(f"  Saturación media: {mean_saturation:.1f}")
        print(f"  Var. Brillo: {std_value:.1f}")
        print(f"  Var. Saturación: {std_saturation:.1f}")
        
        is_very_dark = mean_value < 70
        is_very_uniform = std_value < 20
        is_low_saturation = mean_saturation < 30
        is_uniform_saturation = std_saturation < 15
        
        # Densidad de bordes
        gray_obj = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_obj, 50, 150)
        edge_pixels = np.sum((edges > 0) & (mask_cnt > 0))
        total_pixels = np.sum(mask_cnt > 0)
        edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0
        is_low_edge_density = edge_density < 0.05
        
        print(f"  Densidad de bordes: {edge_density:.3f}")
        
        background_score = 0
        if is_very_dark: 
            background_score += 2
            print("  +2 Puntos: Muy oscuro")
        if is_very_uniform: 
            background_score += 1
            print("  +1 Punto: Muy uniforme")
        if is_low_saturation: 
            background_score += 1
            print("  +1 Punto: Baja saturación")
        if is_uniform_saturation: 
            background_score += 1
            print("  +1 Punto: Saturación uniforme")
        if is_low_edge_density: 
            background_score += 1
            print("  +1 Punto: Pocos bordes")
            
        print(f"  SCORE TOTAL: {background_score}")
        
        if background_score >= 3:
            print("  ⚠️ RECHAZADO POR FILTRO DE FONDO")
        else:
            print("  ✓ ACEPTADO")

