import os, glob, argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# pre-processamento
def preprocess(img_bgr):
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    blur   = cv2.GaussianBlur(clahe, (7, 7), 2)
    edges  = cv2.Canny(blur, 150, 200)
    return gray, clahe, edges

# exibir as etapas de preprocessamento
def show_steps(img_path, pause=True):
    img = cv2.imread(img_path)
    if img is None:
        print(f'[erro] não abriu: {img_path}')
        return

    gray, clahe_img, edges = preprocess(img)

   
    overlay = img.copy()
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:              # ignora ruídos
            (x, y), r = cv2.minEnclosingCircle(c)
            cv2.circle(overlay, (int(x), int(y)), int(r), (0, 255, 0), 2)

    
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].set_title('Original')
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].axis('off')

    ax[1].set_title('Gray + CLAHE')
    ax[1].imshow(clahe_img, cmap='gray'); ax[1].axis('off')

    ax[2].set_title('Canny')
    ax[2].imshow(edges, cmap='gray'); ax[2].axis('off')

    ax[3].set_title('Contorno sobreposto')
    ax[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); ax[3].axis('off')

    plt.tight_layout()
    if pause:
        plt.show(block=True)
    else:
        plt.show(block=False); plt.pause(1); plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Visualiza o pipeline de pré‑processamento das moedas')
    ap.add_argument('--dataset', default='Projeto Yuri/dataset',
                    help='Pasta‑raiz com "1 real" e "50 centavos" (default: %(default)s)')
    ap.add_argument('--single', help='Exibe somente essa imagem')
    ap.add_argument('--nopause', action='store_true',
                    help='Mostra todas as imagens sem pausar (requer fechar janela no final)')
    args = ap.parse_args()

    if args.single:               # visualizar apenas um arquivo
        show_steps(args.single, pause=True)
    else:                         # percorrer todo o dataset
        for cls in ['1 real', '50 centavos']:
            folder = os.path.join(args.dataset, cls)
            for img_path in glob.glob(os.path.join(folder, '*.*')):
                print(f'Pré‑processando {img_path}')
                show_steps(img_path, pause=not args.nopause)
