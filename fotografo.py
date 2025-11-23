#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Captura de imágenes con calibración de cámara (según ost.txt).

- Muestra la imagen original y la corregida (undistort).
- Pulsa 's' para guardar una foto.
- Pulsa 'q' para salir.
"""

import cv2
import os
import time

# ==== Parámetros de la cámara según ost.txt ====
# width  640
# height 480

# [narrow_stereo] camera matrix
# 1284.750848 0.000000 358.790467
# 0.000000 1281.464689 200.809023
# 0.000000 0.000000 1.000000
camera_matrix = [
    [1284.750848,    0.0,        358.790467],
    [   0.0,      1281.464689,   200.809023],
    [   0.0,         0.0,          1.0     ]
]

# [narrow_stereo] distortion
# -0.404441 0.942400 0.001988 -0.002131 0.000000
dist_coeffs = [-0.404441, 0.942400, 0.001988, -0.002131, 0.0]

# Tamaño esperado de la imagen
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# ==== Configuración ====
CAM_INDEX = 1            # Índice de la cámara (0 por defecto)
OUTPUT_DIR = "capturas_buenas"   # Carpeta donde se guardan las imágenes

def main():
    # Crear carpeta de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Convertir listas a matrices de OpenCV
    import numpy as np
    K = np.array(camera_matrix, dtype=np.float32)
    D = np.array(dist_coeffs, dtype=np.float32)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara.")
        return

    # Fijar resolución (por si la cámara lo permite)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    # Obtener newCameraMatrix para undistort
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, D, (IMAGE_WIDTH, IMAGE_HEIGHT), 1, (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    print("Cámara abierta.")
    print("Pulsa 's' para guardar foto, 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: no se pudo leer un frame de la cámara.")
            break

        # Asegurarnos del tamaño esperado
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Imagen corregida (sin distorsión)
        undistorted = cv2.undistort(frame, K, D, None, new_K)

        # Dibujar ambas imágenes
        cv2.imshow("Original", frame)
        cv2.imshow("Corregida (undistort)", undistorted)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Salir
            break
        elif key == ord('s'):
            # Guardar imágenes
            timestamp = int(time.time())
            filename_orig = os.path.join(OUTPUT_DIR, f"orig_{timestamp}.png")
            filename_und = os.path.join(OUTPUT_DIR, f"undist_{timestamp}.png")

            cv2.imwrite(filename_orig, frame)
            cv2.imwrite(filename_und, undistorted)

            print(f"Guardadas: {filename_orig} y {filename_und}")

    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada. Fin del programa.")

if __name__ == "__main__":
    main()
