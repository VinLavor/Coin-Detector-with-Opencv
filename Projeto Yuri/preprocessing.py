import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


img_path = 'Projeto Yuri/moedas_iluminadas.jpeg'
assert os.path.isfile(img_path), f'Arquivo não encontrado: {img_path}'

img_bgr = cv2.imread(img_path)                         # BGR (OpenCV)
gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

clahe      = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
gray_eq    = clahe.apply(gray)

smooth     = cv2.GaussianBlur(gray_eq, (7, 7), sigmaX=2)

edges = cv2.Canny(smooth, 150, 200)

# Hough mais restritivo
circles = cv2.HoughCircles(
    smooth, cv2.HOUGH_GRADIENT,
    dp=1, minDist=120,
    param1=120, param2=45,
    minRadius=50, maxRadius=100
)

# Pós-filtro opcional
if circles is not None:
    filt = []
    for x, y, r in sorted(circles[0], key=lambda c: c[2], reverse=True):
        if all(np.hypot(x-fx, y-fy) > r*0.5 for fx, fy, _ in filt):
            filt.append((x, y, r))
    circles = np.array([filt], dtype=np.uint16)


draw = img_bgr.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f'{len(circles[0])} círculo(s) detectado(s):')
    for i, (x, y, r) in enumerate(circles[0], 1):
        print(f'  #{i}: centro=({x},{y}), raio={r}px')
        # contorno
        cv2.circle(draw, (x, y), r, (0, 255, 0), 2)
        # centro
        cv2.circle(draw, (x, y), 2, (0, 0, 255), 3)
        # índice
        cv2.putText(draw, f'{i}', (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
else:
    print('Nenhum círculo detectado — ajuste param2, minRadius ou thresholds.')

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.title('Original (cinza)')
plt.imshow(gray, cmap='gray'); plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Após CLAHE')
plt.imshow(gray_eq, cmap='gray'); plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Pós Canny')
plt.imshow(edges, cmap='gray'); plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Detecção por Hough')
# converter BGR→RGB para exibir com Matplotlib
plt.imshow(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)); plt.axis('off')

plt.tight_layout(); plt.show()


cv2.imwrite('Projeto Yuri/moedas_hough.jpg', draw)
print('Imagem com círculos salva como moedas_hough.jpg')
