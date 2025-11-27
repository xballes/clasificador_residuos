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

    # ------------------------------------------------------------------ #
    # LATA
    # ------------------------------------------------------------------ #
    def _score_can(self, f: Dict[str, float]) -> float:
        """
        Puntuación para LATA.

        Solo queremos dar buena puntuación a objetos claramente metálicos,
        ni demasiado alargados y con forma razonablemente compacta.
        """
        score = 0.0
    
        area        = f.get("area", 0.0)
        circularity = f.get("circularity", 0.0)
        aspect      = f.get("aspect_ratio", 1.0)
        elong       = f.get("elongation_ratio", 1.0)
        metallic    = f.get("metallic_score", 0.0)
        is_met_col  = f.get("is_metallic_color", 0.0) > 0.5
        spec_top    = f.get("specular_ratio_top", 0.0)
        edge_top    = f.get("edge_density_top", 0.0)
        num_vert    = f.get("num_vertices", 0)

        # 1) Filtro duro: si no es claramente metálico, no queremos lata
        if metallic < 0.45 and not is_met_col:
            return -40.0   # MUCHO más penalizador

        if not is_met_col:
            metallic *= 0.3  # reduce el score si no es color metálico REAL

        # 2) Forma: compacta y no muy alargada
        #    - lata vista desde arriba → circular
        #    - lata de lado → rectángulo corto
        if 0.6 <= circularity <= 1.1 and 0.5 <= aspect <= 2.0 and elong <= 3.0:
            score += 40.0
        elif 0.4 <= circularity <= 0.9 and elong <= 3.5:
            score += 25.0

        # Rectangular (tipo caja metálica)
        if 4 <= num_vert <= 8 and circularity < 0.6 and 0.4 < aspect < 2.5:
            score += 15.0

        # Penalizar cosas muy alargadas (más típico de botella tumbada)
        if elong > 4.0:
            score -= 25.0
        elif elong > 3.0:
            score -= 10.0

        # 3) Metalicidad
        score += metallic * 25.0
        score += spec_top * 10.0
        score += edge_top * 5.0

        # 4) Tamaño típico de lata
        if 1500 <= area <= 25000:
            score += 10.0

        return score

    # ------------------------------------------------------------------ #
    # BOTELLA
    # ------------------------------------------------------------------ #
    def _score_bottle(self, f: Dict[str, float]) -> float:
        """
        Puntuación para BOTELLA.

        Botella =
            - poco metálica
            - bastante alargada (tumbada o de pie)
            - o disco no metálico (vista desde arriba)
        """
        score = 0.0

        area        = f.get("area", 0.0)
        circularity = f.get("circularity", 0.0)
        aspect      = f.get("aspect_ratio", 1.0)
        elong       = f.get("elongation_ratio", 1.0)
        metallic    = f.get("metallic_score", 0.0)
        is_met_col  = f.get("is_metallic_color", 0.0) > 0.5
        is_trans    = f.get("is_transparent_color", 0.0) > 0.5
        sat         = f.get("saturation_mean", 0.0)

        # 1) Preferimos muy poco metal
        if metallic < 0.25 and not is_met_col:
            score += 25.0
        elif metallic < 0.4 and not is_met_col:
            score += 15.0
        else:
            score -= 15.0   # muy metálico → no botella

        # 2) Forma alargada (botella de lado o de pie)
        if elong > 4.0:
            score += 30.0
        elif elong > 3.0:
            score += 25.0
        elif elong > 2.0:
            score += 18.0

        if aspect > 3.0 or aspect < 1 / 3:
            score += 20.0
        elif aspect > 2.0 or aspect < 0.5:
            score += 12.0

        # 3) Botella vista desde arriba: disco NO metálico
        if circularity > 0.8 and metallic < 0.3 and not is_met_col:
            score += 20.0

        # 4) Transparente / poco saturado → plástico típico
        if is_trans:
            score += 15.0
        if sat < 80:
            score += 10.0

        # 5) Tamaño típico de botella (más grande que lata normal)
        if 5000 <= area <= 45000:
            score += 10.0

        return score

    # ------------------------------------------------------------------ #
    # CARTON
    # ------------------------------------------------------------------ #
    def _score_cardboard(self, f: Dict[str, float]) -> float:
        """
        Puntuación para CARTON (brick, caja, etc.).
        """
        score = 0.0

        area        = f.get("area", 0.0)
        circularity = f.get("circularity", 0.0)
        aspect      = f.get("aspect_ratio", 1.0)
        elong       = f.get("elongation_ratio", 1.0)
        metallic    = f.get("metallic_score", 0.0)
        is_met_col  = f.get("is_metallic_color", 0.0) > 0.5
        num_vert    = f.get("num_vertices", 0)
        is_brown    = f.get("is_brown_color", 0.0) > 0.5
        hue         = f.get("hue_mean", 0.0)
        sat         = f.get("saturation_mean", 0.0)
        val         = f.get("value_mean", 0.0)
        tex_var     = f.get("texture_variance", 0.0)
        homo        = f.get("homogeneity", 0.0)
        spec_top    = f.get("specular_ratio_top", 0.0)
        edge_top    = f.get("edge_density_top", 0.0)

        # 1) No metálico casi siempre
        if metallic < 0.25 and not is_met_col:
            score += 20.0
        elif metallic < 0.4 and not is_met_col:
            score += 10.0
        else:
            score -= 20.0

        # 2) Forma rectangular / brick
        if 4 <= num_vert <= 6:
            score += 25.0
        elif 3 <= num_vert <= 8:
            score += 15.0

        if 0.6 <= aspect <= 1.4:
            score += 15.0
        elif 0.4 <= aspect <= 2.0:
            score += 8.0

        if circularity < 0.5:
            score += 20.0
        elif circularity < 0.7:
            score += 10.0
    
        # 4) Textura: algo de textura (no totalmente liso)
        if 50 < tex_var < 1500:
            score += 8.0
        score += (1.0 - homo) * 8.0  # menos homogéneo → más cartón

        # 5) Evitar cosas muy alargadas (eso suelen ser botellas)
        if elong > 4.0:
            score -= 20.0
        elif elong > 3.0:
            score -= 10.0

        # 6) Penalizar tapa metálica tipo lata
        if spec_top > 0.02 and edge_top > 0.02:
            score -= 10.0

        # 7) Tamaño razonable
        if area < 1500:
            score -= 10.0

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
