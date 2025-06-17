# Projeto Prático - Processamento Digital de Imagens

Este projeto foi desenvolvido como atividade da disciplina **SIN 392 - Processamento Digital de Imagens** da Universidade Federal de Viçosa, sob orientação da **Profª Drª Larissa Ferreira Rodrigues Moreira**. Ele permite o carregamento de imagens, visualização de histogramas e aplicação de diversos filtros e transformações, tudo por meio de uma interface simples e funcional.

## Vídeo de Apresentação do Projeto
```bash
https://youtu.be/RFRA3vmDY-4
```

## Funcionalidades

- Upload de imagem (JPG ou PNG)
- Visualização do histograma da imagem original
- Aplicação de filtros espaciais e no domínio da frequência
- Transformações de intensidade
- Operações morfológicas
- Segmentação com Otsu
- Download da imagem processada

## Como executar

1. Clone o repositório

```bash
git clone https://github.com/vinimeirelres/analiseimagens.git
cd analiseimagens
```

2. Crie e ative um ambiente virtual

```bash
python -m venv venv

# No Windows:
venv\Scripts\activate

# No Linux/macOS:
source venv/bin/activate
```

3. Instale as dependências

```bash
pip install -r requirements.txt
```

4. Execute o sistema

```bash
python routes.py
```

5. Acesse no navegador

```
http://localhost:5000
```

## Estrutura do projeto

```
.
├── static/
│   └── css/
│       └── estilo.css
├── templates/
│   ├── index.html
│   ├── processing.html
│   └── histograma.html
├── processamento.py
├── routes.py
├── requirements.txt
└── README.md
```

## Notas

- O backend é feito em Flask.
- O processamento de imagem é realizado com OpenCV.
- O histograma é renderizado no frontend com Chart.js.
