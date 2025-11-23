# Proyecto de Visión Artificial: Clasificación de Residuos

## Estructura del Proyecto

- **`fotografo.py`**: Script para capturar imágenes utilizando una cámara calibrada. Permite guardar imágenes originales y corregidas (undistort).
- **`train_features.py`**: Entrena un modelo de clasificación (Random Forest) basado en características extraídas manualmente (color, forma, textura).
    - Características: Estadísticas de color (HSV), Hu momentos, solidez, circularidad, densidad de bordes, etc.
    - Incluye mejora de imágenes para baja iluminación y características de uniformidad de color.
- **`predict_real.py`**: Script de inferencia.
    - Carga el modelo entrenado (`feature_model.pkl`).
    - Procesa imágenes de la carpeta `capturas/real`.
    - Detecta objetos, extrae características y clasifica.
    - Genera imágenes con las predicciones visualizadas en `output_real`.
- **`capturas/`**: Directorio que contiene las imágenes de entrenamiento y prueba.
    - `botella/`, `carton/`, `lata/`: Imágenes para entrenamiento.
    - `real/`: Imágenes de prueba con múltiples objetos.
  
## Uso
### 1. Captura de Imágenes (Opcional)

Si necesitas capturar nuevas imágenes para entrenamiento:

```bash
python fotografo.py
```
- `s`: Guardar foto.
- `q`: Salir.

### 2. Entrenamiento del Modelo

Para entrenar el modelo con las imágenes en `capturas/`:

```bash
python train_features.py
```
Esto generará `feature_model.pkl` y `feature_scaler.pkl`.

### 3. Predicción

Para clasificar objetos en las imágenes de `capturas/real`:

```bash
python predict_real.py
```
Los resultados se guardarán en la carpeta `output_real`.

## Metodología

El sistema utiliza un enfoque híbrido:
1.  **Preprocesamiento**: Corrección de distorsión de lente, ajuste de gamma para baja iluminación.
2.  **Segmentación**: Umbralización adaptativa y operaciones morfológicas para aislar objetos.
3.  **Extracción de Características**: Se calcula un vector de características para cada objeto (Color, Forma, Textura).
4.  **Clasificación**: Un clasificador Random Forest predice la clase del objeto.
