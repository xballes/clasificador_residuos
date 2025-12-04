# Clasificador de Residuos

Sistema de clasificación automática de residuos (latas, botellas, cartón) usando visión por computadora y machine learning.

## Ejemplos de Uso

### 1. Captura de Imágenes (No hace falta ejecutar)

```bash
python fotografo.py
```

**Controles:** `s` para guardar foto, `q` para salir

---

### 2. Clasificación Manual de Datos (No hace falta ejecutar, ya esta hecho)

```bash
python clean_data.py --source capturas_buenas/real --dest capturas_buenas
```

**Controles:** `L` (lata), `B` (botella), `C` (carton), `S` (skip), `Q` (quit)

---

### 3. Entrenamiento del Modelo ML (No hace falta ejecutar, ya esta hecho)

```bash
python train_classifier.py --data_dir capturas_buenas --output_csv features_extracted.csv
```

---

### 4. Clasificación de Residuos (Estos son los comandos buenos)

#### Imagen Individual

```bash
python clasificador_main.py --input capturas_buenas/real/undist_1764003047.png --output debug_misclassification_v8.png --ml --show-debug --show-roi
```

#### Modo Batch

```bash
python clasificador_main.py --batch capturas_buenas/real --output results/batch_results --show-roi --show-debug --ml
```

#### Sin Machine Learning (clasificador por reglas) ( Esto era como estaba hecho antes, NO ejecutar)

```bash
python clasificador_main.py --input test.png --output result.png --show-roi --show-debug
```

#### Con parámetros personalizados

```bash
python clasificador_main.py --input test.png --output result.png --ml --min-area 800 --confidence 0.4 --roi-margin 30
```

---

## Opciones Principales

**Clasificador (`clasificador_main.py`):**
- `--input`, `-i` : Imagen de entrada
- `--output`, `-o` : Imagen de salida
- `--batch`, `-b` : Carpeta de imágenes (modo batch)
- `--ml` : Usar clasificador ML (requiere entrenamiento previo)
- `--show-roi` : Guardar visualización del ROI
- `--show-debug` : Guardar imágenes de debug
- `--min-area <int>` : Área mínima de objetos (default: 1000)
- `--confidence <float>` : Umbral de confianza (default: 0.35)
- `--no-aruco` : No detectar marcadores ArUco
- `--no-box` : No detectar caja de la izquierda
- `--pattern <str>` : Patrón de archivos para batch (default: *.png)

---
