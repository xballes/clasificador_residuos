import cv2
import numpy as np
from typing import Dict, Tuple, List


class FeatureExtractor:
    """
    Extrae características de objetos detectados para clasificación.
    Características: forma, circularidad, color, metalicidad, textura, tamaño.
    """
    
    def __init__(self):
        pass
    
    def extract_features(self, image: np.ndarray, contour: np.ndarray, 
                        mask: np.ndarray = None) -> Dict[str, float]:
        """
        Extrae todas las características de un objeto dado su contorno.
        
        Args:
            image: Imagen original en BGR
            contour: Contorno del objeto
            mask: Máscara binaria del objeto (opcional)
            
        Returns:
            Diccionario con todas las características extraídas
        """
        features = {}
        
        # Características de forma
        features.update(self._extract_shape_features(contour))
        
        # Características de color
        features.update(self._extract_color_features(image, contour))
        
        # Características de metalicidad
        features.update(self._extract_metallic_features(image, contour))
        
        # Características de textura
        features.update(self._extract_texture_features(image, contour))
        
        return features
    
    def _extract_shape_features(self, contour: np.ndarray) -> Dict[str, float]:
        """Extrae características de forma del contorno."""
        features = {}
        
        # Área y perímetro
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        features['area'] = area
        features['perimeter'] = perimeter
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        features['bbox_width'] = w
        features['bbox_height'] = h
        features['bbox_area'] = w * h
        
        # Aspect ratio (relación de aspecto)
        if h > 0:
            features['aspect_ratio'] = float(w) / h
        else:
            features['aspect_ratio'] = 0.0
        
        # Extent (proporción del área del contorno respecto al bounding box)
        if w * h > 0:
            features['extent'] = area / (w * h)
        else:
            features['extent'] = 0.0
        
        # Circularidad (4π * área / perímetro²)
        # Valor de 1.0 = círculo perfecto
        if perimeter > 0:
            features['circularity'] = (4 * np.pi * area) / (perimeter ** 2)
        else:
            features['circularity'] = 0.0
        
        # Compacidad (área / perímetro)
        if perimeter > 0:
            features['compactness'] = area / perimeter
        else:
            features['compactness'] = 0.0
        
        # Aproximación poligonal para contar vértices
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        features['num_vertices'] = len(approx)
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            features['solidity'] = area / hull_area
        else:
            features['solidity'] = 0.0
        
        # Momentos para orientación
        moments = cv2.moments(contour)
        if moments['m00'] > 0:
            # Centroide
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            features['centroid_x'] = cx
            features['centroid_y'] = cy
        else:
            features['centroid_x'] = 0.0
            features['centroid_y'] = 0.0
        
        return features
    
    def _extract_color_features(self, image: np.ndarray, 
                                contour: np.ndarray) -> Dict[str, float]:
        """Extrae características de color del objeto."""
        features = {}
        
        # Crear máscara del contorno
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Convertir a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calcular valores medios de HSV
        mean_hsv = cv2.mean(hsv, mask=mask)
        features['hue_mean'] = mean_hsv[0]
        features['saturation_mean'] = mean_hsv[1]
        features['value_mean'] = mean_hsv[2]
        
        # Calcular desviación estándar
        std_hsv = np.std(hsv[mask > 0], axis=0)
        features['hue_std'] = std_hsv[0] if len(std_hsv) > 0 else 0.0
        features['saturation_std'] = std_hsv[1] if len(std_hsv) > 1 else 0.0
        features['value_std'] = std_hsv[2] if len(std_hsv) > 2 else 0.0
        
        # Color dominante (histograma)
        hist_h = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        features['dominant_hue'] = np.argmax(hist_h)
        
        # Calcular si es color metálico (plateado/dorado)
        # Plateado: baja saturación, alto valor
        # Dorado: hue ~15-45, saturación media-alta
        is_metallic_color = (mean_hsv[1] < 50 and mean_hsv[2] > 150) or \
                           (15 <= mean_hsv[0] <= 45 and mean_hsv[1] > 100)
        features['is_metallic_color'] = 1.0 if is_metallic_color else 0.0
        
        # Calcular si es transparente/translúcido (bajo valor, baja saturación)
        is_transparent = mean_hsv[1] < 40 and mean_hsv[2] < 100
        features['is_transparent_color'] = 1.0 if is_transparent else 0.0
        
        # Calcular si es marrón/beige (típico de cartón)
        # Marrón: hue ~10-30, saturación baja-media
        is_brown = (10 <= mean_hsv[0] <= 30 and 20 <= mean_hsv[1] <= 150)
        features['is_brown_color'] = 1.0 if is_brown else 0.0
        
        return features
    
    def _extract_metallic_features(self, image: np.ndarray, 
                                   contour: np.ndarray) -> Dict[str, float]:
        """Extrae características de metalicidad (brillo especular, reflexiones)."""
        features = {}
        
        # Crear máscara del contorno
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar píxeles muy brillantes (reflexiones especulares)
        bright_threshold = 200
        bright_pixels = np.sum((gray > bright_threshold) & (mask > 0))
        total_pixels = np.sum(mask > 0)
        
        if total_pixels > 0:
            features['specular_ratio'] = bright_pixels / total_pixels
        else:
            features['specular_ratio'] = 0.0
        
        # Calcular gradiente de intensidad (objetos metálicos tienen gradientes fuertes)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        masked_gradient = gradient_magnitude[mask > 0]
        if len(masked_gradient) > 0:
            features['gradient_mean'] = np.mean(masked_gradient)
            features['gradient_std'] = np.std(masked_gradient)
        else:
            features['gradient_mean'] = 0.0
            features['gradient_std'] = 0.0
        
        # Rango de intensidad (objetos metálicos tienen alto rango)
        masked_gray = gray[mask > 0]
        if len(masked_gray) > 0:
            features['intensity_range'] = np.max(masked_gray) - np.min(masked_gray)
            features['intensity_mean'] = np.mean(masked_gray)
            features['intensity_std'] = np.std(masked_gray)
        else:
            features['intensity_range'] = 0.0
            features['intensity_mean'] = 0.0
            features['intensity_std'] = 0.0
        
        # Score de metalicidad combinado
        metallic_score = (
            features['specular_ratio'] * 0.4 +
            min(features['gradient_mean'] / 100.0, 1.0) * 0.3 +
            min(features['intensity_range'] / 255.0, 1.0) * 0.3
        )
        features['metallic_score'] = metallic_score
        
        return features
    
    def _extract_texture_features(self, image: np.ndarray, 
                                  contour: np.ndarray) -> Dict[str, float]:
        """Extrae características de textura del objeto."""
        features = {}
        
        # Crear máscara del contorno
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Varianza de intensidad (textura rugosa tiene alta varianza)
        masked_gray = gray[mask > 0]
        if len(masked_gray) > 0:
            features['texture_variance'] = np.var(masked_gray)
        else:
            features['texture_variance'] = 0.0
        
        # Detectar bordes internos (textura compleja tiene muchos bordes)
        edges = cv2.Canny(gray, 50, 150)
        internal_edges = edges & mask
        edge_pixels = np.sum(internal_edges > 0)
        total_pixels = np.sum(mask > 0)
        
        if total_pixels > 0:
            features['edge_density'] = edge_pixels / total_pixels
        else:
            features['edge_density'] = 0.0
        
        # Calcular homogeneidad (objetos lisos tienen alta homogeneidad)
        if len(masked_gray) > 0:
            # Usar desviación estándar como medida inversa de homogeneidad
            std = np.std(masked_gray)
            features['homogeneity'] = 1.0 / (1.0 + std / 50.0)
        else:
            features['homogeneity'] = 0.0
        
        return features
    
    def print_features(self, features: Dict[str, float], indent: int = 0):
        """Imprime las características de forma legible."""
        prefix = "  " * indent
        
        print(f"{prefix}=== SHAPE FEATURES ===")
        print(f"{prefix}Area: {features['area']:.1f}")
        print(f"{prefix}Perimeter: {features['perimeter']:.1f}")
        print(f"{prefix}Aspect Ratio: {features['aspect_ratio']:.2f}")
        print(f"{prefix}Circularity: {features['circularity']:.3f}")
        print(f"{prefix}Compactness: {features['compactness']:.2f}")
        print(f"{prefix}Vertices: {features['num_vertices']}")
        print(f"{prefix}Solidity: {features['solidity']:.3f}")
        
        print(f"\n{prefix}=== COLOR FEATURES ===")
        print(f"{prefix}Hue Mean: {features['hue_mean']:.1f}")
        print(f"{prefix}Saturation Mean: {features['saturation_mean']:.1f}")
        print(f"{prefix}Value Mean: {features['value_mean']:.1f}")
        print(f"{prefix}Is Metallic Color: {features['is_metallic_color']}")
        print(f"{prefix}Is Transparent Color: {features['is_transparent_color']}")
        print(f"{prefix}Is Brown Color: {features['is_brown_color']}")
        
        print(f"\n{prefix}=== METALLIC FEATURES ===")
        print(f"{prefix}Specular Ratio: {features['specular_ratio']:.3f}")
        print(f"{prefix}Gradient Mean: {features['gradient_mean']:.2f}")
        print(f"{prefix}Intensity Range: {features['intensity_range']:.1f}")
        print(f"{prefix}Metallic Score: {features['metallic_score']:.3f}")
        
        print(f"\n{prefix}=== TEXTURE FEATURES ===")
        print(f"{prefix}Texture Variance: {features['texture_variance']:.2f}")
        print(f"{prefix}Edge Density: {features['edge_density']:.3f}")
        print(f"{prefix}Homogeneity: {features['homogeneity']:.3f}")
