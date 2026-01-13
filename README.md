# Clasificador de Residuos

Sistema de clasificación automática de residuos (latas, botellas, cartón) usando visión por computadora y machine learning.

## Resumen del Proyecto

Este proyecto implementa un pipeline completo de visión por computador que procesa imágenes en tiempo real para identificar tipos de residuos. El flujo de trabajo consta de cuatro etapas principales:

1.  **Preprocesamiento y ROI**: Calibración de cámara y detección de la zona de trabajo útil, excluyendo elementos distractores como marcadores ArUco o cajas mediante máscaras dinámicas.
2.  **Segmentación Multi-modal**: Separación precisa de los objetos del fondo utilizando una combinación de color (HSV), detección de bordes (Canny) y umbralizado adaptativo.
3.  **Extracción de Características**: Análisis de cada objeto para obtener métricas geométricas (circularidad, relación de aspecto), de color (histogramas HSV) y de textura (brillos especulares para detectar metales).
4.  **Clasificación**:
    *   **ML Classifier (Principal)**: Utiliza modelos de Machine Learning (Random Forest/SVM/KNN) entrenados con características extraídas para una clasificación robusta.

El sistema soporta modos de operación en tiempo real (webcam), procesamiento por lotes y captura estática, ofreciendo visualización de resultados y coordenadas relativas en centímetros.

## Ejemplos de Uso

### 1. Captura de Imágenes (No hace falta ejecutar)

```bash
python fotografo.py
```

**Controles:** `s` para guardar foto, `q` para salir

---



### 2. Entrenamiento del Modelo ML (No hace falta ejecutar, ya esta hecho)

```bash
python train_classifier.py --data_dir capturas_buenas --output_csv features_extracted.csv
```

---

### 3. Clasificación de Residuos (Estos son los comandos buenos)

#### Imagen Individual

```bash
python clasificador_main.py --input capturas_buenas/real/undist_1764003047.png --output debug_misclassification_v8.png --show-debug --show-roi
```

#### Modo Batch

```bash
python clasificador_main.py --batch capturas_buenas/real --output results/batch_results --show-roi --show-debug
```

#### Modo Tiempo Real (Webcam)

```bash
python clasificador_main.py --realtime --camera 0 --show-roi
```

#### Modo Captura (Foto con cuenta atrás)

```bash
python clasificador_main.py --capture --countdown 5 --output captura_nueva.png
```

#### Filtrando por tipo de objeto

```bash
python clasificador_main.py --realtime --filter-class lata
```

#### Sin Machine Learning (clasificador por reglas)

```bash
python clasificador_main.py --input test.png --output result.png --no-ml --show-roi --show-debug
```

#### Con parámetros personalizados

```bash
python clasificador_main.py --input test.png --output result.png --min-area 800 --confidence 0.4 --roi-margin 30 --aruco-size 5.0
```

---

## Opciones Principales

**Clasificador (`clasificador_main.py`):**

**Entrada/Salida:**
- `--input`, `-i` : Ruta de la imagen de entrada.
- `--output`, `-o` : Ruta de la imagen de salida.
- `--batch`, `-b` : Carpeta de imágenes para procesar en lote.
- `--pattern` : Patrón de archivos para modo batch (default: `*.png`).
- `--realtime` : Ejecutar en modo tiempo real con webcam.
- `--capture` : Modo captura: cuenta atrás y captura una imagen.
- `--camera` : Índice de la cámara para modo tiempo real/captura (default: 0).
- `--record` : Grabar video de la sesión en tiempo real.
- `--calib` : Fichero de calibración de cámara (`.yaml`, `.txt`, `.pkl`).

**Configuración de Detección:**
- `--min-area <int>` : Área mínima para detectar objetos (default: 500).
- `--roi-margin <int>` : Margen alrededor de áreas excluidas (default: 10).
- `--board-margin <int>` : Margen interior del tablero negro (default: 0).
- `--confidence <float>` : Umbral de confianza para clasificación (default: 0.35).
- `--filter-class <str>` : Filtrar por tipo: `plastico`, `carton`, `lata`.
- `--aruco-size <float>` : Tamaño real del lado del ArUco en cm (default: 4.8).

**Flags de Control:**
- `--no-ml` : NO usar clasificador ML (usar reglas heurísticas). Por defecto usa ML.
- `--no-aruco` : No detectar marcadores ArUco.
- `--no-box` : No detectar caja de la izquierda.
- `--countdown <int>` : Segundos de cuenta atrás para modo captura (default: 3).

**Visualización y Debug:**
- `--show-roi` : Mostrar/Guardar visualización del ROI.
- `--show-debug` : Guardar imágenes de debug de segmentación.
- `--quiet`, `-q` : Modo silencioso (menos output en consola).
