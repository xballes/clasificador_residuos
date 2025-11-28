import cv2
import os
import shutil
from pathlib import Path
import argparse
import numpy as np
import time

# Importar componentes del sistema
from roi_detector import ROIDetector
from object_segmenter import ObjectSegmenter

def clean_data(source_dir, base_dir):
    """
    Herramienta para clasificar manualmente objetos detectados en imágenes.
    Detecta objetos, los recorta y pide al usuario que los clasifique.
    """
    source_path = Path(source_dir)
    base_path = Path(base_dir)
    
    # Carpetas de destino
    dirs = {
        ord('l'): base_path / 'lata',
        ord('b'): base_path / 'botella',
        ord('c'): base_path / 'carton',
        ord('L'): base_path / 'lata',
        ord('B'): base_path / 'botella',
        ord('C'): base_path / 'carton'
    }
    
    # Crear directorios
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    # Inicializar detectores
    roi_detector = ROIDetector()
    segmenter = ObjectSegmenter(min_area=1000)
    
    # Obtener imágenes
    extensions = ['*.png', '*.jpg', '*.jpeg']
    images = []
    for ext in extensions:
        images.extend(list(source_path.glob(ext)))
        
    print(f"Encontradas {len(images)} imágenes en {source_dir}")
    print("\nControles:")
    print("  [L] - LATA")
    print("  [B] - BOTELLA")
    print("  [C] - CARTON")
    print("  [S] - Saltar objeto")
    print("  [N] - Siguiente imagen (saltar resto de objetos)")
    print("  [Q] - Salir")
    
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] Procesando: {img_path.name}")
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # 1. Detectar ROI
        roi_mask, _ = roi_detector.create_roi_mask(img)
        
        # 2. Segmentar objetos
        contours, _ = segmenter.segment_objects(img, roi_mask=roi_mask)
        
        if not contours:
            print("  No se detectaron objetos.")
            continue
            
        print(f"  Detectados {len(contours)} objetos.")
        
        skip_image = False
        
        for j, cnt in enumerate(contours):
            if skip_image:
                break
                
            # Obtener bounding box y recortar
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Añadir un pequeño margen al recorte
            margin = 10
            h_img, w_img = img.shape[:2]
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w_img, x + w + margin)
            y2 = min(h_img, y + h + margin)
            
            crop = img[y1:y2, x1:x2]
            
            # Visualización: Mostrar imagen completa con recuadro Y el recorte
            vis_full = img.copy()
            cv2.rectangle(vis_full, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis_full, f"Objeto {j+1}/{len(contours)}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Redimensionar full si es muy grande
            if vis_full.shape[0] > 600:
                scale = 600 / vis_full.shape[0]
                vis_full = cv2.resize(vis_full, (0,0), fx=scale, fy=scale)
            
            # Mostrar ventanas
            cv2.imshow("Contexto", vis_full)
            cv2.imshow("OBJETO A CLASIFICAR", crop)
            
            # Esperar tecla
            while True:
                key = cv2.waitKey(0)
                
                if key == ord('q') or key == 27: # Quit
                    cv2.destroyAllWindows()
                    return
                    
                elif key == ord('n') or key == ord('N'): # Next image
                    skip_image = True
                    break
                    
                elif key == ord('s') or key == ord('S'): # Skip object
                    print("  Objeto saltado.")
                    break
                    
                elif key in dirs: # Classify
                    dest_dir = dirs[key]
                    # Nombre único: original_objX.png
                    filename = f"{img_path.stem}_obj{j}.png"
                    dest_path = dest_dir / filename
                    
                    cv2.imwrite(str(dest_path), crop)
                    print(f"  Guardado en {dest_dir.name}: {filename}")
                    break
                    
    cv2.destroyAllWindows()
    print("\nProceso finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Herramienta de clasificación manual de objetos')
    parser.add_argument('--source', default='capturas_buenas/real', help='Carpeta con imágenes originales')
    parser.add_argument('--dest', default='capturas_buenas', help='Carpeta base destino')
    
    args = parser.parse_args()
    
    clean_data(args.source, args.dest)
