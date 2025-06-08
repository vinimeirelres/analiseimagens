from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2 
import numpy as np
from werkzeug.utils import secure_filename
import os
import processamento as p

app = Flask(__name__, template_folder='paginas')

PASTA_UPLOADS = 'uploads'
PASTA_PROCESSADAS = 'processadas'
os.makedirs(PASTA_UPLOADS, exist_ok=True)
os.makedirs(PASTA_PROCESSADAS, exist_ok=True)

app.config['PASTA_UPLOADS'] = PASTA_UPLOADS
app.config['PASTA_PROCESSADAS'] = PASTA_PROCESSADAS

#Tipos de processamento

PROCESSAMENTOS = {
    'alargamento_contraste': lambda img: p.alargamento_contraste(p.escala_cinza(img)),
    'eq_histograma': lambda img: p.equalizacao_histograma(p.escala_cinza(img)),
    'media': lambda img: p.filtro_media(p.escala_cinza(img)),
    'mediana': lambda img: p.filtro_mediana(p.escala_cinza(img)),
    'gaussiano': lambda img: p.filtro_gaussiano(p.escala_cinza(img)),
    'maximo': lambda img: p.filtro_maximo(p.escala_cinza(img)),
    'minimo': lambda img: p.filtro_minimo(p.escala_cinza(img)),
    'laplaciano': lambda img: p.filtro_laplaciano(p.escala_cinza(img)),
    'roberts': lambda img: p.filtro_roberts(p.escala_cinza(img)),
    'prewitt': lambda img: p.filtro_prewitt(p.escala_cinza(img)),
    'sobel': lambda img: p.filtro_sobel(p.escala_cinza(img)),
    'convolucao_dominio': lambda img, tipo: p.aplicar_filtro_frequencia(p.escala_cinza(img), tipo),
    'fourier': lambda img: p.espectro_fourier(p.escala_cinza(img)),
    'erosao': lambda img: p.erosao(p.escala_cinza(img)),
    'dilatacao': lambda img: p.dilatacao(p.escala_cinza(img)),
    'abertura': lambda img: p.abertura(p.escala_cinza(img)),
    'fechamento': lambda img: p.fechamento(p.escala_cinza(img)),
    'otsu': lambda img: p.segmentacao_otsu(p.escala_cinza(img))
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('imagem')
        if not file:
            return 'Nenhuma imagem enviada', 400
        
        filename = secure_filename(file.filename)
        caminho = os.path.join(app.config['PASTA_UPLOADS'], filename)
        file.save(caminho)

        return render_template('processing.html', filename=filename)  # já carrega a próxima página

    return render_template('index.html')



@app.route('/histograma')
def histograma():
    filename = request.args.get('filename')
    if not filename:
        return 'Arquivo não especificado', 400
    return render_template('histograma.html', filename=filename)

@app.route('/processing')
def voltar_para_processing():
    filename = request.args.get('filename')
    if not filename:
        return 'Arquivo não especificado', 400
    return render_template('processing.html', filename=filename)



@app.route('/api/processar', methods=['POST'])
def processar_imagem():
    filename = request.form.get('filename')
    tipo_proc = request.form.get('tipo')
    tipo_freq = request.form.get('tipoFiltro')

    if not filename or not tipo_proc:
        return jsonify({'error': 'Dados ausentes'}), 400

    caminho = os.path.join(app.config['PASTA_UPLOADS'], filename)
    img = cv2.imread(caminho)

    func = PROCESSAMENTOS.get(tipo_proc)
    if not func:
        return jsonify({'error': 'Tipo inválido'}), 400

    try:
        if tipo_proc == 'convolucao_dominio':
            img_proc = func(img, tipo_freq)
        else:
            img_proc = func(img)
    except Exception as e:
        return jsonify({'error': f'Erro: {str(e)}'}), 500

    caminho_processado = os.path.join(app.config['PASTA_PROCESSADAS'], filename)
    cv2.imwrite(caminho_processado, img_proc)

    return jsonify({
        'processada': f'/imagem/processada/{filename}'
    }),200

 
 

@app.route('/api/calcular_histograma', methods=['POST'])
def calcular_histograma():
    filename = request.form.get('filename')
    if not filename:
        return jsonify({'error': 'Nome do arquivo ausente'}), 400
   
    caminho = os.path.join(app.config['PASTA_UPLOADS'], filename)
    img = cv2.imread(caminho)
   
    if img is None:
        return jsonify({'error': 'Imagem não encontrada'}), 404

    hist_data = p.calcular_histograma(p.escala_cinza(img)).flatten().tolist()
 
    
    return jsonify({'histograma': hist_data}),200


@app.route('/imagem/original/<nome>')
def get_original(nome):
    return send_from_directory(app.config['PASTA_UPLOADS'], nome)

@app.route('/imagem/processada/<nome>')
def get_processada(nome):
    return send_from_directory(app.config['PASTA_PROCESSADAS'], nome)


if __name__ == '__main__':
    app.run(debug=True)