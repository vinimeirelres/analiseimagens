import cv2
import numpy as np

# ===========================
# ESCALA DE CINZA
# ===========================
def escala_cinza(imagem):
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


# ===========================
# HISTOGRAMA
# ===========================

def calcular_histograma(imagem):
    hist = cv2.calcHist([imagem], [0], None, [256], [0, 256])
    return hist


# ===========================
# TRANSFORMAÇÕES DE INTENSIDADE
# ===========================

def alargamento_contraste (imagem):
    min_val = np.min(imagem)
    max_val = np.max(imagem)
    alargada = (imagem - min_val) * (255 / (max_val - min_val))

    return alargada.astype(np.uint8)

def equalizacao_histograma(imagem):
    return cv2.equalizeHist(imagem)


# ===========================
# FILTROS PASSA-BAIXA
# ===========================

def filtro_media(imagem):
    return cv2.blur(imagem, (5, 5))


def filtro_mediana(imagem):
    return cv2.medianBlur(imagem, 5)

def filtro_gaussiano(imagem):
    return cv2.GaussianBlur(imagem, (5, 5), 0)

def filtro_maximo(imagem):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(imagem, kernel)

def filtro_minimo(imagem):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(imagem, kernel)


# ===========================
# FILTROS PASSA-ALTA
# ===========================

def filtro_laplaciano(imagem):
    return cv2.Laplacian(imagem, cv2.CV_64F)

def filtro_roberts(imagem):
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    x = cv2.filter2D(imagem, -1, kernel_x)
    y = cv2.filter2D(imagem, -1, kernel_y)
    return cv2.magnitude(x.astype(np.float32), y.astype(np.float32))

def filtro_prewitt(imagem):
    kernelx = np.array([1, 0,-1], [1, 0, -1], [1, 0, -1])
    kernely = np.array([1, 1, 1], [0, 0, 0], [-1, -1, -1])
    x = cv2.filter2D(imagem, -1, kernelx)
    y = cv2.filter2D(imagem, -1, kernely)
    return cv2.magnitude(x.astype(np.float32), y.astype(np.float32))

def filtro_sobel(imagem):
    x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=5)
    y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.magnitude(x.astype(np.float32), y.astype(np.float32))


# ===========================
# CONVOLUÇÃO NO DOMÍNIO DA FREQUÊNCIA
# ===========================

def aplicar_filtro_frequencia(imagem, tipo='passa-baixa'):
    dft = cv2.dft(np.float32(imagem), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = imagem.shape
    crow, ccol = rows//2, cols//2

    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 30
    if tipo == 'passa-baixa':
        mask[crow-r:crow+r, ccol-r:ccol+r] = 1
    elif tipo == 'passa-alta':
        mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ===========================
# ESPECTRO DE FOURIER
# ===========================

def espectro_fourier(imagem):
    dft = cv2.dft(np.float32(imagem), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    return cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ===========================
# MORFOLOGIA MATEMÁTICA
# ===========================

def erosao(imagem):
    kernel = np.ones((5,5), np.uint8)
    return cv2.erode(imagem, kernel)

def dilatacao(imagem):
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(imagem, kernel)

def abertura(imagem):
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(imagem, cv2.MORPH_OPEN, kernel)

def fechamento(imagem):
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, kernel)


# ===========================
# SEGMENTAÇÃO OTSU
# ===========================

def segmentacao_otsu(imagem):
    _, thresh = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
