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
    val_min = np.min(imagem)
    val_max = np.max(imagem)
    alargada = (imagem - val_min) * (255 / (val_max - val_min))

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
    elem_est = np.ones((5, 5), np.uint8)
    return cv2.dilate(imagem, elem_est)

def filtro_minimo(imagem):
    elem_est = np.ones((5, 5), np.uint8)
    return cv2.erode(imagem, elem_est)


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
    kernelx = np.array([[1, 0,-1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
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
    fourier_imagem = cv2.dft(np.float32(imagem), flags=cv2.DFT_COMPLEX_OUTPUT)
    cent_imagem = np.fft.fftshift(fourier_imagem) #centraliza o espectro
    linhas, colunas = imagem.shape
    clinhas, ccolunas = linhas//2, colunas//2

    r = 30  #raio do filtro circular
    if tipo == 'passa-baixa':
        mascara = np.zeros((linhas, colunas, 2), np.uint8)
        mascara[clinhas-r:clinhas+r, ccolunas-r:ccolunas+r] = 1
    elif tipo == 'passa-alta':
        mascara = np.ones((linhas, colunas, 2), np.uint8)
        mascara[clinhas-r:clinhas+r, ccolunas-r:ccolunas+r] = 0

    filtrada = cent_imagem*mascara
    descent_imagem = np.fft.ifftshift(filtrada) #descentraliza o espectro
    imagem_volta= cv2.idft(descent_imagem) #inversa de fourier
    imagem_volta = cv2.magnitude(imagem_volta[:,:,0], imagem_volta[:,:,1]) #calculo da magnitude (eliminação de complexos)

    return cv2.normalize(imagem_volta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #normalização e conversão do tipo


# ===========================
# ESPECTRO DE FOURIER
# ===========================

def espectro_fourier(imagem):
    imagem_fourier = cv2.dft(np.float32(imagem), flags=cv2.DFT_COMPLEX_OUTPUT) #transformada discreta
    cent_freq_imagem = np.fft.fftshift(imagem_fourier) #centraliza a frequência (frequencia zero no centro da imagem, ao invés de no canto superior esquerdo)
    magnitude = 20*np.log(cv2.magnitude(cent_freq_imagem[:,:,0], cent_freq_imagem[:,:,1])) #calculo da magnitude
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #normalização


# ===========================
# MORFOLOGIA MATEMÁTICA
# ===========================

def erosao(imagem):
    elem_est = np.ones((5,5), np.uint8)
    return cv2.erode(imagem, elem_est)

def dilatacao(imagem):
    elem_est = np.ones((5,5), np.uint8)
    return cv2.dilate(imagem, elem_est)

def abertura(imagem):
    elem_est = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(imagem, cv2.MORPH_OPEN, elem_est)

def fechamento(imagem):
    elem_est = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, elem_est)


# ===========================
# SEGMENTAÇÃO OTSU
# ===========================

def segmentacao_otsu(imagem):
    limiar_otimo_otsu, imagem_limiarizada = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(limiar_otimo_otsu) #limiar ótimo calculado pelo método de Otsu
    return imagem_limiarizada
