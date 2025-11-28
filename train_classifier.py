import os
import cv2
import numpy as np
import pandas as pd
import joblib
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Importar el extractor de características existente
from feature_extractor import FeatureExtractor

def get_segmentation_mask(image):
    """
    Obtiene la máscara del objeto usando la lógica robusta del usuario.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check corners to determine background color
    h, w = gray.shape
    corners = [
        gray[0:10, 0:10],
        gray[0:10, w-10:w],
        gray[h-10:h, 0:10],
        gray[h-10:h, w-10:w]
    ]
    avg_corner_brightness = np.mean([np.mean(c) for c in corners])
    
    # Determine threshold type based on background
    if avg_corner_brightness > 127:
        # Light background -> Object is darker -> Invert for mask
        thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        # Dark background -> Object is lighter
        thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        
    # Segmentation
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, thresh_type)
    
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def load_data_from_images(data_dir):
    """
    Carga imágenes y extrae SOLO las features ML.
    """
    print(f"Cargando imágenes desde: {data_dir}")
    
    extractor = FeatureExtractor()
    data = []
    
    class_map = {
        'lata': 'LATA',
        'botella': 'BOTELLA',
        'carton': 'CARTON'
    }
    
    for folder_name, class_name in class_map.items():
        folder_path = Path(data_dir) / folder_name
        if not folder_path.exists():
            print(f"Advertencia: No se encontró la carpeta {folder_path}")
            continue
            
        print(f"Procesando clase: {class_name} ({folder_name})...")
        images = list(folder_path.glob('*.png')) + list(folder_path.glob('*.jpg'))
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Usar segmentación robusta
            mask = get_segmentation_mask(img)
            
            # Encontrar contorno en la máscara
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) < 500:
                continue
                
            try:
                # Extraer TODAS las features
                all_features = extractor.extract_features(img, largest_contour, mask=mask)
                
                # Filtrar SOLO las features que empiezan por 'ml_'
                ml_features = {k: v for k, v in all_features.items() if k.startswith('ml_')}
                
                if not ml_features:
                    print(f"Advertencia: No se extrajeron features ML para {img_path.name}")
                    continue
                
                ml_features['class'] = class_name
                ml_features['image_name'] = img_path.name
                data.append(ml_features)
                
            except Exception as e:
                print(f"Error procesando {img_path.name}: {e}")
                
    return pd.DataFrame(data)

def train_models(df, output_dir='models'):
    """Entrena modelos y guarda el mejor."""
    if df.empty:
        print("No hay datos para entrenar.")
        return
        
    print(f"\nTotal de muestras: {len(df)}")
    print(df['class'].value_counts())
    
    # Preparar datos
    X = df.drop(['class', 'image_name'], axis=1, errors='ignore')
    # Convertir a float explícitamente y llenar NaNs
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    y = df['class']
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    # Modelos a probar
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVC': SVC(probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_score = 0
    best_model = None
    best_name = ""
    
    print("\nEntrenando modelos...")
    
    for name, model in models.items():
        # Cross validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_cv = cv_scores.mean()
        
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"\n{name}:")
        print(f"  CV Score: {mean_cv:.4f}")
        print(f"  Test Score: {test_score:.4f}")
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name
            
    print(f"\nMejor modelo: {best_name} con score {best_score:.4f}")
    
    # Evaluación detallada del mejor modelo
    y_pred = best_model.predict(X_test)
    print("\nReporte de Clasificación (Test Set):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Guardar
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(output_dir, 'waste_classifier_model.pkl')
    joblib.dump(best_model, model_path)
    
    # Guardar scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Guardar encoder
    encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    joblib.dump(le, encoder_path)
    
    # Guardar nombres de columnas
    columns_path = os.path.join(output_dir, 'feature_columns.pkl')
    joblib.dump(X.columns.tolist(), columns_path)
    
    print(f"\nModelo guardado en {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Entrenar clasificador de residuos')
    parser.add_argument('--data_dir', type=str, default='capturas_buenas', 
                        help='Directorio con carpetas de imágenes (lata, botella, carton)')
    parser.add_argument('--output_csv', type=str, default='features_extracted.csv',
                        help='Archivo CSV donde guardar las features extraídas (opcional)')
    
    args = parser.parse_args()
    
    # 1. Cargar y extraer
    df = load_data_from_images(args.data_dir)
    
    if not df.empty and args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Features guardadas en {args.output_csv}")
        
    # 2. Entrenar
    train_models(df)

if __name__ == "__main__":
    main()
