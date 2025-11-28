#!/usr/bin/env python3
"""
Clasificador de Residuos - Script Principal

Detecta y clasifica objetos (latas, botellas, cartón) en imágenes.
Excluye marcadores ArUco y cajas usando ROI.

Uso:
    python clasificador_main.py --input <imagen> --output <salida> [opciones]
    python clasificador_main.py --batch <carpeta> --output <carpeta_salida>
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from typing import List, Tuple

from feature_extractor import FeatureExtractor
from roi_detector import ROIDetector
from object_segmenter import ObjectSegmenter
from waste_classifier import WasteClassifier
from ml_waste_classifier import MLWasteClassifier


class WasteClassificationSystem:
    """Sistema completo de clasificación de residuos."""
    
    def __init__(self, 
                 min_area: int = 1000,
                 roi_margin: int = 20,
                 confidence_threshold: float = 0.4,
                 detect_aruco: bool = True,
                 detect_box: bool = True,
                 use_ml: bool = False):
        """
        Args:
            min_area: Área mínima para detectar objetos
            roi_margin: Margen alrededor de áreas excluidas
            confidence_threshold: Umbral de confianza para clasificación
            detect_aruco: Si True, detecta y excluye ArUco markers
            detect_box: Si True, detecta y excluye caja de la izquierda
            use_ml: Si True, usa el clasificador ML
        """
        self.feature_extractor = FeatureExtractor()
        self.roi_detector = ROIDetector(margin=roi_margin)
        self.segmenter = ObjectSegmenter(min_area=min_area)
        
        if use_ml:
            print("Usando Clasificador ML...")
            self.classifier = MLWasteClassifier(confidence_threshold=confidence_threshold)
        else:
            print("Usando Clasificador por Reglas...")
            self.classifier = WasteClassifier(confidence_threshold=confidence_threshold)
        
        self.detect_aruco = detect_aruco
        self.detect_box = detect_box
    
    def process_image(self, 
                     image_path: str,
                     output_path: str = None,
                     show_roi: bool = False,
                     show_debug: bool = False,
                     verbose: bool = True) -> dict:
        """
        Procesa una imagen completa: detecta, segmenta y clasifica objetos.
        
        Args:
            image_path: Ruta de la imagen de entrada
            output_path: Ruta de la imagen de salida (opcional)
            show_roi: Si True, visualiza el ROI
            show_debug: Si True, guarda imágenes de debug
            verbose: Si True, imprime información detallada
            
        Returns:
            Diccionario con resultados de clasificación
        """
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Procesando: {image_path}")
            print(f"Tamaño de imagen: {image.shape}")
            print(f"{'='*60}\n")
        
        # 1. Crear ROI mask
        if verbose:
            print("Paso 1: Detectando ROI...")
        
        roi_mask, roi_info = self.roi_detector.create_roi_mask(
            image, 
            detect_aruco=self.detect_aruco,
            detect_box=self.detect_box
        )
        
        if verbose:
            print(f"  - ArUco markers detectados: {len(roi_info['aruco_markers'])}")
            print(f"  - Caja detectada: {'Sí' if roi_info['box_region'] else 'No'}")
            print(f"  - Áreas excluidas: {roi_info['excluded_areas']}")
        
        # 2. Segmentar objetos
        if verbose:
            print("\nPaso 2: Segmentando objetos...")
        
        contours, seg_info = self.segmenter.segment_objects(
            image, 
            roi_mask=roi_mask,
            debug=show_debug
        )
        
        if verbose:
            print(f"  - Contornos totales: {seg_info['num_contours_total']}")
            print(f"  - Contornos válidos: {seg_info['num_contours_valid']}")
        
        # 3. Extraer características y clasificar cada objeto
        if verbose:
            print("\nPaso 3: Clasificando objetos...")
        
        results = []
        for i, contour in enumerate(contours):
            if verbose:
                print(f"\n  Objeto #{i+1}:")
            
            # Extraer características
            features = self.feature_extractor.extract_features(image, contour)
            
            # Clasificar
            class_name, confidence, scores = self.classifier.classify(features)
            
            # Guardar resultado
            result = {
                'id': i + 1,
                'contour': contour,
                'features': features,
                'class': class_name,
                'confidence': confidence,
                'scores': scores
            }
            results.append(result)
            
            if verbose:
                print("    metallic_score =", features.get("metallic_score"))
                print("    specular_ratio =", features.get("specular_ratio"))
                print("    specular_ratio_top =", features.get("specular_ratio_top"))
                print("    gradient_mean =", features.get("gradient_mean"))
                print("    is_metallic_color =", features.get("is_metallic_color"))
                print("    is_transparent_color =", features.get("is_transparent_color"))
                print("    is_brown_color =", features.get("is_brown_color"))
                print("    circularity =", features.get("circularity"))
                print("    elongation_ratio =", features.get("elongation_ratio"))
                print("    edge_density_top =", features.get("edge_density_top"))

        
        # 4. Crear visualización
        vis_image = self._create_visualization(image, results)
        
        # 5. Guardar resultados
        if output_path:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            cv2.imwrite(output_path, vis_image)
            if verbose:
                print(f"\n✓ Imagen guardada en: {output_path}")
            
            # Guardar visualización de ROI si se solicita
            if show_roi:
                roi_vis = self.roi_detector.visualize_roi(image, roi_mask, roi_info)
                roi_path = output_path.replace('.png', '_roi.png')
                cv2.imwrite(roi_path, roi_vis)
                if verbose:
                    print(f"✓ ROI guardado en: {roi_path}")
            
            # Guardar debug de segmentación si se solicita
            if show_debug:
                debug_vis = self.segmenter.create_debug_visualization(seg_info)
                debug_path = output_path.replace('.png', '_debug.png')
                cv2.imwrite(debug_path, debug_vis)
                if verbose:
                    print(f"✓ Debug guardado en: {debug_path}")
        
        # Resumen
        if verbose:
            print(f"\n{'='*60}")
            print("RESUMEN DE CLASIFICACIÓN:")
            class_counts = {}
            for result in results:
                cls = result['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            for cls, count in class_counts.items():
                print(f"  {cls}: {count}")
            print(f"{'='*60}\n")
        
        return {
            'image_path': image_path,
            'output_path': output_path,
            'num_objects': len(results),
            'results': results,
            'roi_info': roi_info,
            'visualization': vis_image
        }
    
    def _create_visualization(self, image: np.ndarray, results: List[dict]) -> np.ndarray:
        """Crea visualización con bounding boxes y etiquetas."""
        vis = image.copy()
        
        for result in results:
            contour = result['contour']
            class_name = result['class']
            confidence = result['confidence']
            obj_id = result['id']
            
            # Color según la clase
            color = self.classifier.get_class_color(class_name)
            
            # Dibujar contorno
            cv2.drawContours(vis, [contour], -1, color, 2)
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            
            # Etiqueta
            label = f"#{obj_id} {class_name} ({confidence:.0%})"
            
            # Fondo para el texto
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x, y-text_h-10), (x+text_w, y), color, -1)
            
            # Texto
            cv2.putText(vis, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Contador total
        total_text = f"Total objetos: {len(results)}"
        cv2.putText(vis, total_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return vis
    
    def process_batch(self, 
                     input_dir: str,
                     output_dir: str,
                     pattern: str = "*.png",
                     **kwargs) -> dict:
        """
        Procesa un lote de imágenes.
        
        Args:
            input_dir: Directorio de entrada
            output_dir: Directorio de salida
            pattern: Patrón de archivos (ej: "*.png", "*.jpg")
            **kwargs: Argumentos adicionales para process_image
            
        Returns:
            Diccionario con estadísticas del lote
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Encontrar todas las imágenes
        image_files = list(input_path.glob(pattern))
        
        print(f"\n{'='*60}")
        print(f"PROCESAMIENTO POR LOTES")
        print(f"Directorio de entrada: {input_dir}")
        print(f"Directorio de salida: {output_dir}")
        print(f"Imágenes encontradas: {len(image_files)}")
        print(f"{'='*60}\n")
        
        all_results = []
        class_totals = {}
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Procesando: {img_file.name}")
            
            output_file = output_path / img_file.name
            
            try:
                result = self.process_image(
                    str(img_file),
                    str(output_file),
                    verbose=False,
                    **kwargs
                )
                
                all_results.append(result)
                
                # Actualizar totales
                for obj_result in result['results']:
                    cls = obj_result['class']
                    class_totals[cls] = class_totals.get(cls, 0) + 1
                
                print(f"  ✓ Objetos detectados: {result['num_objects']}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # Resumen final
        print(f"\n{'='*60}")
        print("RESUMEN DEL LOTE:")
        print(f"  Imágenes procesadas: {len(all_results)}/{len(image_files)}")
        print(f"  Total de objetos: {sum(class_totals.values())}")
        print(f"\n  Distribución por clase:")
        for cls, count in sorted(class_totals.items()):
            print(f"    {cls}: {count}")
        print(f"{'='*60}\n")
        
        return {
            'num_images': len(image_files),
            'num_processed': len(all_results),
            'class_totals': class_totals,
            'results': all_results
        }


def main():
    """Función principal con interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Clasificador de Residuos - Detecta y clasifica latas, botellas y cartón'
    )
    
    # Argumentos de entrada/salida
    parser.add_argument('--input', '-i', type=str,
                       help='Ruta de la imagen de entrada')
    parser.add_argument('--output', '-o', type=str,
                       help='Ruta de la imagen de salida')
    parser.add_argument('--batch', '-b', type=str,
                       help='Procesar todas las imágenes en un directorio')
    parser.add_argument('--pattern', type=str, default='*.png',
                       help='Patrón de archivos para modo batch (default: *.png)')
    
    # Opciones de visualización
    parser.add_argument('--show-roi', action='store_true',
                       help='Mostrar visualización del ROI')
    parser.add_argument('--show-debug', action='store_true',
                       help='Guardar imágenes de debug de segmentación')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Modo silencioso (menos output)')
    
    
    # Opciones de configuración
    parser.add_argument('--min-area', type=int, default=1000,
                       help='Área mínima para detectar objetos (default: 1000)')
    parser.add_argument('--roi-margin', type=int, default=20,
                       help='Margen alrededor de áreas excluidas (default: 20)')
    parser.add_argument('--confidence', type=float, default=0.35,
                       help='Umbral de confianza (default: 0.35)')
    parser.add_argument('--no-aruco', action='store_true',
                       help='No detectar marcadores ArUco')
    parser.add_argument('--no-box', action='store_true',
                       help='No detectar caja de la izquierda')
    parser.add_argument('--ml', action='store_true',
                       help='Usar clasificador ML (requiere haber ejecutado train_classifier.py)')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.input and not args.batch:
        parser.error('Debe especificar --input o --batch')
    
    if args.input and not args.output:
        # Generar nombre de salida automático
        args.output = args.input.replace('.png', '_classified.png')
    
    # Crear sistema de clasificación
    system = WasteClassificationSystem(
        min_area=args.min_area,
        roi_margin=args.roi_margin,
        confidence_threshold=args.confidence,
        detect_aruco=not args.no_aruco,
        detect_box=not args.no_box,
        use_ml=args.ml
    )
    
    # Procesar
    if args.batch:
        # Modo batch
        system.process_batch(
            args.batch,
            args.output,
            pattern=args.pattern,
            show_roi=args.show_roi,
            show_debug=args.show_debug
        )
    else:
        # Modo single image
        system.process_image(
            args.input,
            args.output,
            show_roi=args.show_roi,
            show_debug=args.show_debug,
            verbose=not args.quiet
        )


if __name__ == '__main__':
    main()