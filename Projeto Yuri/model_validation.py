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

def validate_preprocessing(img_path, display=True, save_steps=False, save_dir="steps"):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Erro ao carregar imagem: {img_path}")

        output_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        
        edges = cv2.Canny(blurred, 50, 150)

        
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("Nenhum contorno encontrado")

        
        largest_contour = max(contours, key=cv2.contourArea)

        
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        x, y, radius = int(x), int(y), int(radius)

        
        x1 = max(x - radius, 0)
        y1 = max(y - radius, 0)
        x2 = min(x + radius, img.shape[1])
        y2 = min(y + radius, img.shape[0])
        roi = img[y1:y2, x1:x2]

        if display or save_steps:
            debug_img = output_img.copy()
            cv2.circle(debug_img, (x, y), radius, (0, 255, 0), 2)     # verde
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # vermelho


            
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0].set_title('Original')
            axs[1].imshow(gray, cmap='gray')
            axs[1].set_title('Cinza')
            axs[2].imshow(edges, cmap='gray')
            axs[2].set_title('Canny')
            axs[3].imshow(closed, cmap='gray')
            axs[3].set_title('Fechamento')
            axs[4].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            axs[4].set_title('Contorno Detectado')

            for ax in axs:
                ax.axis('off')
            plt.tight_layout()

            if save_steps:
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(save_dir, f"{filename}_steps.png")
                plt.savefig(save_path)
                print(f"Etapas salvas em: {save_path}")

            if display:
                plt.show()
            else:
                plt.close()

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


val_path = "Projeto Yuri/dataset/val"
model_dir = "Projeto Yuri/models"

# Carrega modelos
svm_model = load(os.path.join(model_dir, "svm_model.joblib"))
rf_model = load(os.path.join(model_dir, "rf_model.joblib"))
mlp_model = load(os.path.join(model_dir, "mlp_model.joblib"))
scaler = load(os.path.join(model_dir, "scaler.joblib"))

# Classes
class_names = ["1real", "50centavos", "25centavos"]

# Coleta dados de validação
X_val, y_val = [], []

for label, class_name in enumerate(class_names):
    class_dir = os.path.join(val_path, class_name)
    for fname in os.listdir(class_dir):
        img_path = os.path.join(class_dir, fname)

        roi = validate_preprocessing(img_path, display=False)
        if roi is None:
            continue

        features = extract_features(roi)
        if features is None:
            continue

        X_val.append(features)
        y_val.append(label)

X_val = np.array(X_val)
y_val = np.array(y_val)

# Normalização
X_val_scaled = scaler.transform(X_val)

# Avaliação
def evaluate(model, name):
    y_pred = model.predict(X_val_scaled)
    print(f"\nAvaliação do modelo {name}")
    print(classification_report(y_val, y_pred, target_names=class_names))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_val, y_pred))

evaluate(svm_model, "SVM")
evaluate(rf_model, "Random Forest")
