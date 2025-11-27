import os
import glob
import csv
import cv2
import numpy as np

from feature_extractor import FeatureExtractor

# Rutas a tus carpetas
DATA_ROOT = "capturas_buenas"
CLASS_FOLDERS = {
    "LATA":    os.path.join(DATA_ROOT, "lata"),
    "CARTON":  os.path.join(DATA_ROOT, "carton"),
    "BOTELLA": os.path.join(DATA_ROOT, "botella"),
}

# ---------------------------------------------------------------------
# Función muy sencilla para obtener el contorno principal de la imagen
# (si tú ya tienes un paso de segmentación bueno, usa el tuyo en su lugar)
# ---------------------------------------------------------------------
def get_main_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Suavizado ligero
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Umbral automático Otsu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # A veces queda invertido (fondo blanco / objeto negro), nos quedamos con la máscara
    # que tenga menos área blanca (muy chapucero, pero suele valer para un dataset controlado)
    white_ratio = np.mean(thresh == 255)
    if white_ratio > 0.5:
        thresh = 255 - thresh

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Contorno más grande = objeto principal
    main_contour = max(contours, key=cv2.contourArea)
    return main_contour

# ---------------------------------------------------------------------
# Recorremos las carpetas y guardamos features en un CSV
# ---------------------------------------------------------------------
def main():
    fe = FeatureExtractor()
    rows = []
    feature_names = None

    for class_name, folder in CLASS_FOLDERS.items():
        image_paths = sorted(
            glob.glob(os.path.join(folder, "*.jpg")) +
            glob.glob(os.path.join(folder, "*.png")) +
            glob.glob(os.path.join(folder, "*.jpeg"))
        )

        print(f"Procesando clase {class_name} con {len(image_paths)} imágenes...")

        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARN] No se pudo leer {img_path}")
                continue

            contour = get_main_contour(image)
            if contour is None or cv2.contourArea(contour) < 100:
                print(f"[WARN] No contorno válido en {img_path}")
                continue

            features = fe.extract_features(image, contour)

            # Inicializamos lista de nombres de features la primera vez
            if feature_names is None:
                feature_names = sorted(features.keys())

            row = {
                "class": class_name,
                "image": os.path.basename(img_path),
            }
            for k in feature_names:
                row[k] = features.get(k, 0.0)
            rows.append(row)

    if not rows:
        print("No se han extraído features. Revisa rutas / segmentación.")
        return

    # Guardar a CSV
    out_csv = "features_capturas_buenas.csv"
    fieldnames = ["class", "image"] + feature_names

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nFeatures guardadas en {out_csv}")

    # -----------------------------------------------------------------
    # Estadísticas rápidas por clase (media y std)
    # -----------------------------------------------------------------
    import collections
    per_class = collections.defaultdict(list)
    for r in rows:
        per_class[r["class"]].append(r)

    print("\n=== ESTADÍSTICAS POR CLASE ===")
    for cls, items in per_class.items():
        print(f"\n--- {cls} ---")
        # Convertimos a array
        arr = np.array([[float(it[f]) for f in feature_names] for it in items])
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)

        for name, m, s in zip(feature_names, mean, std):
            print(f"{name:25s}  mean={m:8.3f}  std={s:8.3f}")


if __name__ == "__main__":
    main()
