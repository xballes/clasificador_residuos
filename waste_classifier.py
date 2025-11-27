import numpy as np
from typing import Dict, Tuple


class WasteClassifier:
    """
    Clasificador de residuos basado en reglas sencillas.

    Clases:
        - LATA
        - BOTELLA
        - CARTON

    La idea es:
        1) Separar objetos METÁLICOS de NO METÁLICOS.
        2) En los NO METÁLICOS, distinguir BOTELLA (alargada / tapa circular)
           de CARTON (rectangular tipo brick).
    """

    CLASS_CAN = "LATA"
    CLASS_BOTTLE = "BOTELLA"
    CLASS_CARDBOARD = "CARTON"
    CLASS_UNKNOWN = "DESCONOCIDO"

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: mínimo de confianza para aceptar una clase.
        """
        self.confidence_threshold = confidence_threshold

    def classify(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Clasifica un objeto a partir de sus características.

        Devuelve:
            (clase_predicha, confianza [0,1], diccionario_scores_por_clase)
        """
        scores = {
            self.CLASS_CAN:        self._score_can(features),
            self.CLASS_BOTTLE:     self._score_bottle(features),
            self.CLASS_CARDBOARD:  self._score_cardboard(features),
        }

        # Elegir mejor clase
        best_class = max(scores, key=scores.get)
        raw_vals = np.array(list(scores.values()), dtype=float)

        # Normalizamos a [0,1] para sacar algo tipo "probabilidad"
        raw_vals -= raw_vals.min()
        total = raw_vals.sum()
        if total <= 0:
            confidence = 0.0
        else:
            probs = raw_vals / total
            confidence = float(probs[list(scores.keys()).index(best_class)])

        # Si la confianza es baja, devolvemos DESCONOCIDO
        if confidence < self.confidence_threshold:
            return self.CLASS_UNKNOWN, confidence, scores

        return best_class, confidence, scores

       # ================================
    # LATA
    # ================================
    def _score_can(self, f: Dict[str, float]) -> float:
        metallic = f.get("metallic_score", 0.0)
        circ     = f.get("circularity", 0.0)
        elong    = f.get("elongation_ratio", 1.0)
        aspect   = f.get("aspect_ratio", 1.0)
        edge_top = f.get("edge_density_top", 0.0)

        score = 0.0

        # --- Lata circular (Aquarius) ---
        # Círculo bastante perfecto, muy metálico, muchos bordes en la tapa
        if metallic > 0.55 and circ > 0.80 and edge_top > 0.09:
            score += 5.0

        # --- Lata rectangular (Smints) ---
        # Rectangular metálica, no extremadamente alargada
        if metallic > 0.55 and 1.3 <= elong <= 4.0 and aspect > 1.2:
            score += 4.0

        # Penalizar cosas exageradamente largas (eso es botella)
        if elong > 4.0:
            score -= 3.0

        return score

    # ================================
    # BOTELLA
    # ================================
    def _score_bottle(self, f: Dict[str, float]) -> float:
        metallic = f.get("metallic_score", 0.0)
        circ     = f.get("circularity", 0.0)
        elong    = f.get("elongation_ratio", 1.0)
        aspect   = f.get("aspect_ratio", 1.0)
        edge_top = f.get("edge_density_top", 0.0)

        score = 0.0

        # --- Botella tumbada: muy alargada ---
        if elong > 2.2 or aspect > 2.2 or aspect < 1.0 / 2.2:
            score += 5.0

        # --- Botella de pie: círculo NO tan metálico y sin agujero ---
        if circ > 0.80 and metallic < 0.55 and edge_top < 0.09:
            score += 4.0

        # Si es círculo muy metálico con muchos bordes, casi seguro es lata
        if circ > 0.80 and metallic > 0.65 and edge_top > 0.09:
            score -= 4.0

        return score

    # ================================
    # CARTON
    # ================================
    def _score_cardboard(self, f: Dict[str, float]) -> float:
        metallic = f.get("metallic_score", 0.0)
        circ     = f.get("circularity", 0.0)
        elong    = f.get("elongation_ratio", 1.0)
        aspect   = f.get("aspect_ratio", 1.0)

        score = 0.0

        # --- Forma cuadrada: tus bricks verde/blanco ---
        if 0.70 <= aspect <= 1.30 and elong < 2.0 and circ < 0.80:
            score += 5.0

        # No exigimos metalicidad baja (porque el cartón verde brilla mucho),
        # solo penalizamos si parece claramente un círculo (tipo lata arriba).
        if circ > 0.80:
            score -= 3.0

        return score




    # ------------------------------------------------------------------ #
    # Utilidades visualización
    # ------------------------------------------------------------------ #
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Color BGR para dibujar cada clase."""
        colors = {
            self.CLASS_CAN:       (0, 255, 255),   # Amarillo
            self.CLASS_BOTTLE:    (255, 0, 0),     # Azul
            self.CLASS_CARDBOARD: (0, 165, 255),   # Naranja
            self.CLASS_UNKNOWN:   (128, 128, 128)  # Gris
        }
        return colors.get(class_name, (255, 255, 255))

    def print_classification(
        self,
        class_name: str,
        confidence: float,
        scores: Dict[str, float],
        indent: int = 0,
    ):
        prefix = "  " * indent
        print(f"{prefix}=== CLASSIFICATION RESULT ===")
        print(f"{prefix}Predicted Class: {class_name}")
        print(f"{prefix}Confidence: {confidence:.2%}")
        print(f"{prefix}Scores:")
        for cls, s in scores.items():
            print(f"{prefix}  {cls}: {s:.2f}")
