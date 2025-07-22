import os
import cv2
import numpy as np
from joblib import load
from skimage.feature import hog


VALIDATION_DIR = "Projeto Yuri/dataset/validation"  
MODELS_DIR = "Projeto Yuri/models"  
CLASS_NAMES = ["1real", "50centavos", "25centavos"] 

def load_models():
    """Carrega os modelos e o scaler"""
    try:
        svm_model = load(os.path.join(MODELS_DIR, "svm_model.joblib"))
        rf_model = load(os.path.join(MODELS_DIR, "rf_model.joblib"))
        mlp_model = load(os.path.join(MODELS_DIR, "mlp_model.joblib"))
        scaler = load(os.path.join(MODELS_DIR, "scaler.joblib"))
        return svm_model, rf_model, mlp_model, scaler
    except Exception as e:
        print(f"Erro ao carregar modelos: {str(e)}")
        return None, None, None, None

def preprocess_and_extract_features(img_path, scaler, fixed_size=(64, 64)):
    """Pré-processa e extrai features de uma imagem"""
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
            raise ValueError("Nenhum contorno encontrado")
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, fixed_size)
        
        features = []
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        features.extend([np.mean(hsv[:,:,0]), np.std(hsv[:,:,0]),
                        np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])])
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray_roi, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(1, 1))
        features.extend(hog_features)
        features.append(float(roi.shape[1]) / roi.shape[0])
        
        return scaler.transform(np.array(features).reshape(1, -1))
    
    except Exception as e:
        print(f"Erro em {os.path.basename(img_path)}: {str(e)}")
        return None

def test_validation_set():
    """Testa todas as imagens na pasta de validação"""
    if not os.path.exists(VALIDATION_DIR):
        print(f"Erro: Pasta de validação não encontrada em '{VALIDATION_DIR}'")
        return
    
    svm_model, rf_model, mlp_model, scaler = load_models()
    if svm_model is None:
        return
    
    results = []
    print(f"\n=== Testando imagens em '{VALIDATION_DIR}' ===")
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(VALIDATION_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Pasta de classe '{class_name}' não encontrada!")
            continue
            
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                features = preprocess_and_extract_features(img_path, scaler)
                
                if features is not None:
                    svm_pred = CLASS_NAMES[svm_model.predict(features)[0]]
                    rf_pred = CLASS_NAMES[rf_model.predict(features)[0]]
                    mlp_pred = CLASS_NAMES[mlp_model.predict(features)[0]]
                    
                    results.append({
                        "Classe Real": class_name,
                        "Imagem": img_file,
                        "SVM": svm_pred,
                        "Random Forest": rf_pred,
                        "MLP": mlp_pred,
                    })

    
    def calculate_accuracy(model_name):
        correct = sum(1 for r in results if r[model_name] == r["Classe Real"])
        return correct, len(results), (correct / len(results) * 100) if results else 0
    
    svm_correct, total, svm_acc = calculate_accuracy("SVM")
    rf_correct, _, rf_acc = calculate_accuracy("Random Forest")
    mlp_correct, _, mlp_acc = calculate_accuracy("MLP")

    
    print("\n=== Resultados Finais ===")
    print(f"\nTotal de imagens testadas: {total}")
    print(f"\nAcurácia dos Modelos:")
    print(f"- SVM: {svm_correct}/{total} ({svm_acc:.1f}%)")
    print(f"- Random Forest: {rf_correct}/{total} ({rf_acc:.1f}%)")
    print(f"- MLP: {mlp_correct}/{total} ({mlp_acc:.1f}%)")

    
    print("\n=== Detalhes por Imagem ===")
    for result in results:
        print(f"\nImagem: {result['Imagem']} | Classe Real: {result['Classe Real']}")
        print(f"SVM: {result['SVM']} {'✅' if result['SVM'] == result['Classe Real'] else '❌'}")
        print(f"Random Forest: {result['Random Forest']} {'✅' if result['Random Forest'] == result['Classe Real'] else '❌'}")
        print(f"MLP: {result['MLP']} {'✅' if result['MLP'] == result['Classe Real'] else '❌'}")

if __name__ == "__main__":
    test_validation_set()