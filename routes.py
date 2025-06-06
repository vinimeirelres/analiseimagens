from flask import Flask, render_template, request, jsonify
import cv2 # OpenCV para processamento de imagem
import numpy as np
import base64
import re # Para extrair dados do Data URL
import processamento as p

app = Flask(__name__, template_folder='paginas')

def data_url_to_cv2_img(data_url):
    """Converte um Data URL de imagem para um objeto de imagem OpenCV."""
    try:
        # Remove o cabeçalho do Data URL (ex: "data:image/png;base64,")
        img_str = re.search(r'base64,(.*)', data_url).group(1)
        img_bytes = base64.b64decode(img_str)
        img_np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Erro ao converter Data URL para imagem: {e}")
        return None

def cv2_img_to_data_url(img, extension=".png"):
    """Converte um objeto de imagem OpenCV para um Data URL."""
    try:
        _, buffer = cv2.imencode(extension, img)
        img_bytes = buffer.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/{extension[1:]};base64,{img_base64}"
    except Exception as e:
        print(f"Erro ao converter imagem para Data URL: {e}")
        return None

@app.route('/')
def carregar_pagina():
    """Serve a página inicial para carregar a imagem."""
    return render_template('index.html') 

@app.route('/processamento')
def processamento():
    """Serve a página que vai exibir a imagem original e a processada."""
    return render_template('processing.html') 

@app.route('/histograma')
def carregar_histograma():
    return render_template('histograma.html') 

@app.route('/api/processar', methods=['POST'])
def api_processar_imagem():
    """
    API para processar uma imagem.
    Recebe: JSON com 'imageDataUrl' e 'tipoProcessamento'.
    Retorna: JSON com 'processedImageDataUrl' ou um erro.
    """
    data = request.get_json()
    if not data or 'imageDataUrl' not in data or 'tipoProcessamento' not in data:
        return jsonify({"error": "Dados inválidos"}), 400

    original_data_url = data['imageDataUrl']
    tipo_processamento = data['tipoProcessamento']

    img_original = data_url_to_cv2_img(original_data_url)
    if img_original is None:
        return jsonify({"error": "Não foi possível decodificar a imagem original"}), 500

    img_processada = None
    # --- Lógica de Processamento ---
    try:
        if tipo_processamento == 'alargamento_contraste':
            if img_original is not None:
                cinza = p.escala_cinza(img_original)
                img_processada = p.alargamento_contraste(cinza)
        elif tipo_processamento == 'media':
            if img_original is not None:
                cinza = p.escala_cinza(img_original)
                img_processada = p.filtro_media(cinza)
        elif tipo_processamento == 'equalizacao_histograma':
            if img_original is not None:
                cinza = p.escala_cinza(img_original)
                img_processada = p.equalizacao_histograma(cinza)
        elif tipo_processamento == 'mediana':
            if img_original is not None:
                cinza = p.escala_cinza(img_original)
                img_processada = p.filtro_mediana(cinza)
        elif tipo_processamento == 'gaussiano':
            if img_original is not None:
                cinza = p.escala_cinza(img_original)
                img_processada = p.filtro_gaussiano(cinza)
        elif tipo_processamento == 'maximo':
            if img_original is not None:
                cinza = p.escala_cinza(img_original)
                img_processada = p.filtro_maximo(cinza)
        elif tipo_processamento == 'minimo':
            if img_original is not None:
                cinza = p.escala_cinza(img_original)
                img_processada = p.filtro_minimo(cinza)        
        # Adicione outros tipos de processamento aqui
        # elif tipo_processamento == 'outro_filtro':
        #     # Seu código de processamento
        #     pass
        else:
            return jsonify({"error": "Tipo de processamento desconhecido"}), 400
    except Exception as e:
        print(f"Erro durante o processamento '{tipo_processamento}': {e}")
        return jsonify({"error": f"Erro ao processar a imagem: {str(e)}"}), 500


    if img_processada is None:
        return jsonify({"error": "Falha no processamento da imagem"}), 500

    processed_data_url = cv2_img_to_data_url(img_processada)
    if processed_data_url is None:
        return jsonify({"error": "Não foi possível codificar a imagem processada"}), 500

    return jsonify({"processedImageDataUrl": processed_data_url})


@app.route('/api/calcular_histograma', methods=['POST'])
def api_calcular_histograma():
    """API para calcular os dados do histograma de uma imagem."""
    data = request.get_json()
    if not data or 'imageDataUrl' not in data:
        return jsonify({"error": "Dados inválidos: imageDataUrl ausente"}), 400

    img = data_url_to_cv2_img(data['imageDataUrl'])
    if img is None:
        return jsonify({"error": "Não foi possível decodificar a imagem"}), 500
    
    # 1. Converter para escala de cinza
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Calcular o histograma
    # Parâmetros: [imagem], [canal], máscara (nenhuma), [tamanho do hist], [range]
    hist = cv2.calcHist([cinza], [0], None, [256], [0, 256])
    
    # 3. Preparar dados para JSON
    # O hist é um array 2D, então usamos flatten() para torná-lo 1D
    # e tolist() para converter de numpy array para uma lista Python padrão
    hist_data = hist.flatten().tolist()
    
    return jsonify({"histogramData": hist_data})


if __name__ == '__main__':
    # Instale o OpenCV se ainda não tiver: pip install opencv-python
    app.run(debug=True)