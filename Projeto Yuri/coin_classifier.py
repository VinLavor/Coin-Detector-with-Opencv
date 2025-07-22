import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt
from joblib import dump, load

def validate_preprocessing(img_path, display=False):

    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Erro ao carregar imagem: {img_path}")
        
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(blur)
        
        
        thresh = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("Nenhum contorno encontrado na imagem")
        
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        roi = img[y:y+h, x:x+w]
        
        if display:
            plt.figure(figsize=(15, 5))
            plt.subplot(141), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
            plt.subplot(142), plt.imshow(thresh, cmap='gray'), plt.title('Threshold')
            plt.subplot(143), plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)), plt.title('ROI')
            
            
            contour_img = img.copy()
            cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 2)
            plt.subplot(144), plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)), plt.title('Contorno')
            plt.show()
        
        return roi
    
    except Exception as e:
        print(f"Erro no pré-processamento da imagem {img_path}: {str(e)}")
        return None

def extract_features(roi, display=False, fixed_size=(64, 64)):
    """Extrai features com tamanho consistente"""
    try:
        
        roi = cv2.resize(roi, fixed_size)
        
        features = []
        
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        features.extend([np.mean(hsv[:,:,0]), np.std(hsv[:,:,0]), 
                        np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])]) 
        
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray_roi, orientations=8, 
                          pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1))
        features.extend(hog_features)  # 128 features (8*(64/16)*(64/16))
        
        
        features.append(float(roi.shape[1]) / roi.shape[0])  
        
        return np.array(features)
    
    except Exception as e:
        print(f"Erro na extração de features: {str(e)}")
        return None

def load_dataset(dataset_path, sample_display=3):
    """Carrega dataset com validação visual"""
    data = []
    labels = []
    class_names = ["1real", "50centavos", "25centavos"]
    stats = {cls: 0 for cls in class_names}
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Diretório não encontrado: {class_dir}")
        
        print(f"\nProcessando classe: {class_name}")
        sample_count = 0
        
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            
            
            display = sample_count < sample_display
            if display:
                print(f"\nVisualizando amostra {sample_count + 1} de {class_name}")
            
            roi = validate_preprocessing(img_path, display=display)
            if roi is None:
                continue
                
            features = extract_features(roi, display=display)
            if features is None:
                continue
                
            data.append(features)
            labels.append(label)
            stats[class_name] += 1
            sample_count += 1
    
    
    print("\n=== Estatísticas do Dataset ===")
    for cls, count in stats.items():
        print(f"{cls}: {count} amostras")
    
    if len(data) == 0:
        raise ValueError("Nenhuma imagem válida foi processada. Verifique seu dataset.")
    
    return np.array(data), np.array(labels)

def train_and_evaluate(X, y):
    """Treina modelos com validação cruzada"""
    try:
        # Validação dos dados
        if len(np.unique(y)) < 2:
            raise ValueError("Dataset precisa ter pelo menos 2 classes válidas")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Normalização
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Modelos
        models = {
            "SVM": SVC(kernel='rbf', C=1.0, gamma='scale'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), 
                                activation='relu', 
                                solver='adam', 
                                max_iter=500, 
                                random_state=42,
                                early_stopping=True)
        }
        
        for name, model in models.items():
            print(f"\n=== Treinando {name} ===")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            print(f"\nRelatório de Classificação - {name}:")
            print(classification_report(y_test, pred, 
                                      target_names=["1real", "50centavos", "25centavos"]))
            
            
            cm = confusion_matrix(y_test, pred)
            print("Matriz de Confusão:")
            print(cm)
    
    except Exception as e:
        print(f"Erro no treinamento: {str(e)}")
        return None, None, None
    
    return models["SVM"], models["Random Forest"], models["MLP"], scaler


def save_models(svm_model, rf_model, mlp_model, scaler, save_dir="Projeto Yuri/models"):
    """Salva os modelos e o scaler em arquivos .joblib"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        dump(svm_model, os.path.join(save_dir, "svm_model.joblib"))
        dump(rf_model, os.path.join(save_dir, "rf_model.joblib"))
        dump(mlp_model, os.path.join(save_dir, "mlp_model.joblib"))
        dump(scaler, os.path.join(save_dir, "scaler.joblib"))
        print(f"\nModelos salvos em '{save_dir}'!")
    except Exception as e:
        print(f"Erro ao salvar modelos: {str(e)}")

if __name__ == "__main__":
    try:
        dataset_path = "Projeto Yuri/dataset"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Diretório do dataset não encontrado: {dataset_path}")
        
        print("=== Processando dataset ===")
        X, y = load_dataset(dataset_path)
        
        print("\n=== Treinando modelos ===")
        svm_model, rf_model, mlp_model, scaler = train_and_evaluate(X, y)
        
        if svm_model is not None:
            save_models(svm_model, rf_model, mlp_model, scaler)  
            print("\nModelos salvos woohoooo!")
            
    except Exception as e:
        print(f"Erro no pipeline eita: {str(e)}")

