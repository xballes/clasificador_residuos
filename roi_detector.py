import cv2
import numpy as np
from typing import Tuple, List, Optional


class ROIDetector:
    """
    Detecta y excluye áreas no deseadas de la imagen:
    - Marcadores ArUco en las esquinas
    - Caja de la izquierda
    Genera una máscara ROI válida para detección de objetos.
    """
    
    def __init__(self, margin: int = 20):
        """
        Args:
            margin: Margen adicional alrededor de áreas excluidas (píxeles)
        """
        self.margin = margin
        
        # Inicializar detector de ArUco - probar múltiples diccionarios
        self.aruco_dicts = [
            cv2.aruco.DICT_4X4_50,
            cv2.aruco.DICT_4X4_100,
            cv2.aruco.DICT_4X4_250,
            cv2.aruco.DICT_5X5_50,
            cv2.aruco.DICT_6X6_50,
            cv2.aruco.DICT_6X6_250
        ]
    
    def create_roi_mask(self, image: np.ndarray, 
                       detect_aruco: bool = True,
                       detect_box: bool = True,
                       debug: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Crea una máscara ROI excluyendo áreas no deseadas.
        
        Args:
            image: Imagen de entrada en BGR
            detect_aruco: Si True, detecta y excluye marcadores ArUco
            detect_box: Si True, detecta y excluye la caja de la izquierda
            debug: Si True, retorna información de debug
            
        Returns:
            Tupla (mask, info) donde:
            - mask: Máscara binaria (255 = ROI válido, 0 = excluido)
            - info: Diccionario con información de detección
        """
        h, w = image.shape[:2]
        
        # Inicializar máscara (todo válido)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        info = {
            'aruco_markers': [],
            'box_region': None,
            'excluded_areas': 0
        }
        
        # Detectar y excluir ArUco markers
        if detect_aruco:
            aruco_markers = self._detect_aruco_markers(image)
            info['aruco_markers'] = aruco_markers
            
            for corners in aruco_markers:
                # Expandir el área del marcador con margen
                expanded_corners = self._expand_polygon(corners, self.margin)
                cv2.fillPoly(mask, [expanded_corners], 0)
                info['excluded_areas'] += 1
        
        # Detectar y excluir caja de la izquierda
        if detect_box:
            box_region = self._detect_left_box(image)
            if box_region is not None:
                info['box_region'] = box_region
                x, y, w_box, h_box = box_region
                # Aplicar margen
                x = max(0, x - self.margin)
                y = max(0, y - self.margin)
                w_box = min(w - x, w_box + 2 * self.margin)
                h_box = min(h - y, h_box + 2 * self.margin)
                
                mask[y:y+h_box, x:x+w_box] = 0
                info['excluded_areas'] += 1
        
        return mask, info
    
    def _detect_aruco_markers(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detecta marcadores ArUco en la imagen.
        Prueba múltiples diccionarios para asegurar detección.
        
        Returns:
            Lista de esquinas de marcadores detectados
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        all_markers = []
        
        # Probar múltiples diccionarios
        for dict_type in self.aruco_dicts:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            
            corners, ids, rejected = detector.detectMarkers(gray)
            
            if corners is not None and len(corners) > 0:
                for corner in corners:
                    # corner tiene forma (1, 4, 2), convertir a (4, 2)
                    corner_points = corner.reshape(-1, 2).astype(np.int32)
                    
                    # Evitar duplicados
                    is_duplicate = False
                    for existing in all_markers:
                        if np.allclose(corner_points, existing, atol=5):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        all_markers.append(corner_points)
        
        return all_markers
    
    def _detect_left_box(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detecta la caja de colores en el lado izquierdo de la imagen.
        
        Estrategia:
        1. Buscar en el lado izquierdo (hasta 30% del ancho)
        2. Detectar objetos con colores saturados (la caja tiene colores vivos)
        3. Filtrar por tamaño y posición
        
        Returns:
            Tupla (x, y, w, h) del bounding box de la caja, o None si no se detecta
        """
        h, w = image.shape[:2]
        
        # Región de interés: 30% izquierdo de la imagen
        left_region_width = int(w * 0.3)
        roi = image[:, :left_region_width].copy()
        
        # Convertir a HSV para detección de colores
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Detectar píxeles con alta saturación (colores vivos de la caja)
        # La caja tiene colores azul, rosa/rojo, amarillo
        mask_saturated = cv2.inRange(hsv[:, :, 1], 80, 255)
        
        # Operaciones morfológicas para limpiar
        kernel = np.ones((5, 5), np.uint8)
        mask_saturated = cv2.morphologyEx(mask_saturated, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_saturated = cv2.morphologyEx(mask_saturated, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask_saturated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Buscar el contorno más grande en la izquierda
        min_area = (h * w) * 0.02  # Al menos 2% del área total
        max_area = (h * w) * 0.25  # Máximo 25% del área total
        
        best_box = None
        best_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filtrar por área
            if area < min_area or area > max_area:
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            
            # Debe estar en el lado izquierdo (x pequeño)
            if x < left_region_width * 0.5 and area > best_area:
                best_area = area
                best_box = (x, y, w_box, h_box)
        
        return best_box
    
    def _expand_polygon(self, points: np.ndarray, margin: int) -> np.ndarray:
        """
        Expande un polígono añadiendo un margen.
        
        Args:
            points: Puntos del polígono (N, 2)
            margin: Margen en píxeles
            
        Returns:
            Puntos expandidos
        """
        # Calcular centroide
        centroid = np.mean(points, axis=0)
        
        # Expandir cada punto alejándolo del centroide
        expanded = []
        for point in points:
            direction = point - centroid
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                new_point = point + direction * margin
                expanded.append(new_point)
            else:
                expanded.append(point)
        
        return np.array(expanded, dtype=np.int32)
    
    def visualize_roi(self, image: np.ndarray, mask: np.ndarray, 
                     info: dict) -> np.ndarray:
        """
        Crea una visualización del ROI y áreas excluidas.
        
        Args:
            image: Imagen original
            mask: Máscara ROI
            info: Información de detección
            
        Returns:
            Imagen con visualización
        """
        vis = image.copy()
        
        # Crear overlay semitransparente para áreas excluidas
        overlay = vis.copy()
        overlay[mask == 0] = [0, 0, 255]  # Rojo para áreas excluidas
        
        # Mezclar con transparencia
        alpha = 0.3
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
        
        # Dibujar contornos de ArUco markers
        for corners in info['aruco_markers']:
            cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
            # Etiquetar
            center = np.mean(corners, axis=0).astype(int)
            cv2.putText(vis, "ArUco", tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Dibujar caja
        if info['box_region'] is not None:
            x, y, w, h = info['box_region']
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(vis, "Box", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Añadir texto informativo
        text = f"Excluded areas: {info['excluded_areas']}"
        cv2.putText(vis, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
    
    def filter_contours_by_roi(self, contours: List[np.ndarray], 
                               mask: np.ndarray,
                               min_overlap: float = 0.5) -> List[np.ndarray]:
        """
        Filtra contornos que están fuera del ROI válido.
        
        Args:
            contours: Lista de contornos
            mask: Máscara ROI (255 = válido, 0 = excluido)
            min_overlap: Proporción mínima del contorno que debe estar en ROI
            
        Returns:
            Lista de contornos válidos
        """
        valid_contours = []
        
        for cnt in contours:
            # Crear máscara del contorno
            cnt_mask = np.zeros_like(mask)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            
            # Calcular overlap con ROI
            overlap = cv2.bitwise_and(cnt_mask, mask)
            overlap_area = np.sum(overlap > 0)
            cnt_area = np.sum(cnt_mask > 0)
            
            if cnt_area > 0:
                overlap_ratio = overlap_area / cnt_area
                
                if overlap_ratio >= min_overlap:
                    valid_contours.append(cnt)
        
        return valid_contours
