import os
import cv2
import numpy as np
from joblib import load
from skimage.feature import hog
import matplotlib.pyplot as plt


def validate_preprocessing(img_path, display=True):
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

        if display:
            debug_img = output_img.copy()
            cv2.circle(debug_img, (x, y), radius, (0, 255, 0), 2)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Imagem Original")
            axs[1].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            axs[1].set_title("ROI detectada")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

        return roi

    except Exception as e:
        print(f"Erro no pré-processamento: {str(e)}")
        return None


def extract_features(roi, fixed_size=(64, 64)):
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
        features.extend(hog_features)

        features.append(float(roi.shape[1]) / roi.shape[0])

        return np.array(features)
    
    except Exception as e:
        print(f"Erro na extração de features: {str(e)}")
        return None

# caso da pilha Projeto Yuri/dataset/val/1real/WhatsApp Image 2025-07-22 at 16.51.13.jpeg
img_path = "Projeto Yuri/dataset/val/1real/WhatsApp Image 2025-07-22 at 16.51.00.jpeg"  #Projeto Yuri/dataset/val/25centavos/WhatsApp Image 2025-07-22 at 17.02.54(1).jpeg
model_dir = "Projeto Yuri/models"


svm_model = load(os.path.join(model_dir, "svm_model.joblib"))
rf_model = load(os.path.join(model_dir, "rf_model.joblib"))
scaler = load(os.path.join(model_dir, "scaler.joblib"))


class_names = ["1real", "50centavos", "25centavos"]


roi = validate_preprocessing(img_path, display=True)
if roi is not None:
    features = extract_features(roi)
    if features is not None:
        features_scaled = scaler.transform([features])

        
        for model, name in zip([svm_model, rf_model],
                               ["SVM", "Random Forest"]):
            pred = model.predict(features_scaled)[0]
            print(f"[{name}] Predição: {class_names[pred]}")
    else:
        print("Erro na extração de features.")
else:
    print("Erro no pré-processamento da imagem.")
