import numpy as np
from typing import Dict, Tuple, Optional


class WasteClassifier:
    """
    Clasificador de residuos basado en características extraídas.
    Clasifica objetos en: LATA, BOTELLA, CARTON
    
    Usa un sistema de puntuación basado en reglas para cada clase.
    """
    
    # Clases de objetos
    CLASS_CAN = "LATA"
    CLASS_BOTTLE = "BOTELLA"
    CLASS_CARDBOARD = "CARTON"
    CLASS_UNKNOWN = "DESCONOCIDO"
    
    def __init__(self, confidence_threshold: float = 0.4):
        """
        Args:
            confidence_threshold: Umbral mínimo de confianza para clasificar
        """
        self.confidence_threshold = confidence_threshold
    
    def classify(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Clasifica un objeto basándose en sus características.
        
        Args:
            features: Diccionario de características extraídas
            
        Returns:
            Tupla (clase, confianza, scores) donde:
            - clase: Clase predicha (LATA, BOTELLA, CARTON, DESCONOCIDO)
            - confianza: Confianza de la predicción [0, 1]
            - scores: Puntuaciones para cada clase
        """
        # Calcular puntuación para cada clase
        scores = {
            self.CLASS_CAN: self._score_can(features),
            self.CLASS_BOTTLE: self._score_bottle(features),
            self.CLASS_CARDBOARD: self._score_cardboard(features)
        }
        
        # Encontrar la clase con mayor puntuación
        best_class = max(scores, key=scores.get)
        best_score = scores[best_class]
        
        # Normalizar scores a [0, 1]
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = best_score / total_score
        else:
            confidence = 0.0
        
        # Verificar umbral de confianza
        if confidence < self.confidence_threshold:
            return self.CLASS_UNKNOWN, confidence, scores
        
        return best_class, confidence, scores
    
    def _score_can(self, f: Dict[str, float]) -> float:
        """
        Calcula puntuación para LATA.
        
        Características de latas:
        - CILÍNDRICAS (Aquarius): Alta circularidad, forma compacta
        - RECTANGULARES (Smints): Baja circularidad, 4-6 vértices, forma rectangular
        - Metalicidad moderada-alta
        - Tamaño pequeño-mediano
        - Colores vivos (amarillo, rojo, azul, verde) o metálicos
        """
        score = 0.0
        
        # Detectar tipo de lata: cilíndrica vs rectangular
        circularity = f.get('circularity', 0.0)
        num_vertices = f.get('num_vertices', 0)
        aspect_ratio = f.get('aspect_ratio', 0.0)
        
        is_cylindrical = circularity > 0.5
        is_rectangular = (num_vertices >= 4 and num_vertices <= 8 and 
                         circularity < 0.4 and 
                         0.3 < aspect_ratio < 3.0)
        
        # Forma (peso: 20%)
        if is_cylindrical:
            # Lata cilíndrica vista desde arriba
            if circularity > 0.7:
                score += 20.0
            elif circularity > 0.5:
                score += 15.0
        elif is_rectangular:
            # Lata rectangular (Smints)
            score += 18.0
        else:
            # Forma intermedia (lata cilíndrica en ángulo)
            if circularity > 0.3:
                score += 12.0
            elif circularity > 0.15:
                score += 8.0
        
        # Metalicidad (peso: 18%)
        metallic_score = f.get('metallic_score', 0.0)
        score += metallic_score * 18.0
        
        # Color vivo o metálico (peso: 25%)
        hue = f.get('hue_mean', 0.0)
        saturation = f.get('saturation_mean', 0.0)
        
        # Amarillo (típico de latas): hue ~20-40
        if 15 <= hue <= 45 and saturation > 100:
            score += 25.0
        # Rojo (latas de Coca-Cola, etc): hue ~0-10 o ~170-180
        elif (hue <= 10 or hue >= 170) and saturation > 100:
            score += 20.0
        # Verde/Cyan (latas de Sprite, Heineken, Aquarius): hue ~80-110
        elif 75 <= hue <= 115 and saturation > 40:
            score += 22.0
        # Azul (latas de Pepsi, etc): hue ~110-140
        elif 110 <= hue <= 140 and saturation > 80:
            score += 20.0
        # Color metálico
        elif f.get('is_metallic_color', 0.0) > 0.5:
            score += 15.0
        # Cualquier color con saturación moderada-alta (latas de colores)
        elif saturation > 60:
            score += 12.0
        
        # Reflexiones especulares (peso: 12%)
        specular_ratio = f.get('specular_ratio', 0.0)
        score += specular_ratio * 12.0
        
        # Tamaño (peso: 15%) - ajustado para ambos tipos de latas
        area = f.get('area', 0.0)
        if 1000 < area < 25000:  # Tamaño típico de lata
            score += 15.0
        elif 500 < area < 35000:
            score += 10.0
        
        # Compacidad (peso: 10%) - latas son objetos compactos
        compactness = f.get('compactness', 0.0)
        if compactness > 15:
            score += 10.0
        elif compactness > 10:
            score += 7.0
        elif compactness > 5:
            score += 4.0
        
        return score
    
    def _score_bottle(self, f: Dict[str, float]) -> float:
        """
        Calcula puntuación para BOTELLA.
        
        Características de botellas:
        - Forma alargada (aspect ratio alto o bajo dependiendo de orientación)
        - Baja metalicidad
        - Transparencia/translucidez (colores claros, baja saturación)
        - Tamaño mediano-grande
        - Baja circularidad
        - NO colores vivos (amarillo, rojo, azul saturado)
        - NO forma compacta (las latas son más compactas)
        """
        score = 0.0
        
        # Forma alargada (peso: 30%) - AUMENTADO
        aspect_ratio = f.get('aspect_ratio', 0.0)
        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            score += 30.0
        elif aspect_ratio > 2.5 or aspect_ratio < 0.4:
            score += 25.0
        elif aspect_ratio > 2.0 or aspect_ratio < 0.5:
            score += 20.0
        elif aspect_ratio > 1.5 or aspect_ratio < 0.67:
            score += 12.0
        
        # Baja metalicidad (peso: 20%)
        metallic_score = f.get('metallic_score', 0.0)
        score += (1.0 - metallic_score) * 20.0
        
        # Transparencia/translucidez (peso: 25%)
        saturation = f.get('saturation_mean', 0.0)
        hue = f.get('hue_mean', 0.0)
        
        # Botellas suelen tener baja saturación
        if saturation < 50:
            score += 15.0
        elif saturation < 80:
            score += 8.0
        
        # Color transparente
        if f.get('is_transparent_color', 0.0) > 0.5:
            score += 10.0
        
        # Penalizar colores vivos (típicos de latas) - AUMENTADO
        if saturation > 120:
            score -= 15.0  # Aumentado de 10 a 15
        elif saturation > 90:
            score -= 8.0
        
        # Baja circularidad (peso: 10%)
        circularity = f.get('circularity', 0.0)
        if circularity < 0.3:
            score += 10.0
        elif circularity < 0.5:
            score += 6.0
        elif circularity < 0.6:
            score += 3.0
        
        # Penalizar alta circularidad (típico de latas)
        if circularity > 0.6:
            score -= 10.0
        
        # Tamaño mediano-grande (peso: 10%)
        area = f.get('area', 0.0)
        if 8000 < area < 35000:
            score += 10.0
        elif 5000 < area < 45000:
            score += 5.0
        
        # Penalizar tamaño muy pequeño (típico de latas)
        if area < 5000:
            score -= 10.0
        
        # Homogeneidad (botellas suelen ser lisas) (peso: 5%)
        homogeneity = f.get('homogeneity', 0.0)
        score += homogeneity * 5.0
        
        # Penalizar alta compacidad (típico de latas)
        compactness = f.get('compactness', 0.0)
        if compactness > 15:
            score -= 10.0
        
        return score
    
    def _score_cardboard(self, f: Dict[str, float]) -> float:
        """
        Calcula puntuación para CARTON.
        
        Características de cartón:
        - Baja circularidad
        - Textura mate (baja metalicidad)
        - Colores marrones/beige
        - Forma rectangular (4-6 vértices)
        - Baja homogeneidad (textura visible)
        """
        score = 0.0
        
        # Forma rectangular (peso: 25%)
        num_vertices = f.get('num_vertices', 0)
        if 4 <= num_vertices <= 6:
            score += 25.0
        elif 3 <= num_vertices <= 8:
            score += 15.0
        
        # Baja circularidad (peso: 20%)
        circularity = f.get('circularity', 0.0)
        if circularity < 0.4:
            score += 20.0
        elif circularity < 0.5:
            score += 15.0
        elif circularity < 0.6:
            score += 10.0
        
        # Color marrón/beige (peso: 25%)
        if f.get('is_brown_color', 0.0) > 0.5:
            score += 25.0
        else:
            # Verificar manualmente
            hue = f.get('hue_mean', 0.0)
            saturation = f.get('saturation_mean', 0.0)
            if 10 <= hue <= 30 and 20 <= saturation <= 150:
                score += 15.0
        
        # Baja metalicidad (peso: 15%)
        metallic_score = f.get('metallic_score', 0.0)
        score += (1.0 - metallic_score) * 15.0
        
        # Textura (cartón tiene textura visible) (peso: 10%)
        texture_variance = f.get('texture_variance', 0.0)
        if 100 < texture_variance < 1000:
            score += 10.0
        elif 50 < texture_variance < 1500:
            score += 5.0
        
        # Baja homogeneidad (peso: 5%)
        homogeneity = f.get('homogeneity', 0.0)
        score += (1.0 - homogeneity) * 5.0
        
        return score
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Retorna el color BGR para visualización de cada clase.
        
        Args:
            class_name: Nombre de la clase
            
        Returns:
            Tupla (B, G, R) con valores 0-255
        """
        colors = {
            self.CLASS_CAN: (0, 255, 255),      # Amarillo (metálico)
            self.CLASS_BOTTLE: (255, 0, 0),     # Azul (transparente)
            self.CLASS_CARDBOARD: (0, 165, 255), # Naranja (cartón)
            self.CLASS_UNKNOWN: (128, 128, 128)  # Gris
        }
        return colors.get(class_name, (255, 255, 255))
    
    def print_classification(self, class_name: str, confidence: float, 
                           scores: Dict[str, float], indent: int = 0):
        """
        Imprime el resultado de clasificación de forma legible.
        """
        prefix = "  " * indent
        
        print(f"{prefix}=== CLASSIFICATION RESULT ===")
        print(f"{prefix}Predicted Class: {class_name}")
        print(f"{prefix}Confidence: {confidence:.2%}")
        print(f"{prefix}")
        print(f"{prefix}Scores by class:")
        for cls, score in scores.items():
            print(f"{prefix}  {cls}: {score:.2f}")
