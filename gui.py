import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import processamento as p

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Projeto Prático - Edição e Análise de Imagens")
        self.root.geometry("1000x600")

        #Inicializa imagem original e processada
        self.img_original = None
        self.img_processada = None

        #Área para exibir a imagem
        self.label_imagem = tk.Label(root)
        self.label_imagem.pack(pady=10)

        #Botões Básicos
        self.btn_carregar = tk.Button(root, text="Carregar Imagem", command=self.carregar_imagem)
        self.btn_carregar.pack(pady=5)

        self.btn_histograma = tk.Button(root, text="Histograma", command=self.mostrar_histograma)
        self.btn_histograma.pack(pady=5)

        self.btn_alargamento = tk.Button(root, text="Alargamento de Contraste", command=self.alargamento_contraste)
        self.btn_alargamento.pack(pady=5)

        self.btn_eqhist = tk.Button(root, text="Equalização de Histograma", command=self.eq_hist)
        self.btn_eqhist.pack(pady=5)

        self.btn_filtromedia = tk.Button(root, text="Filtro da Média", command=self.filtro_media)
        self.btn_filtromedia.pack(pady=5)

        self.btn_salvar = tk.Button(root, text="Salvar Imagem", command=self.salvar_imagem)
        self.btn_salvar.pack(pady=5)

    def carregar_imagem(self):
        caminho_imagem = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg;*.jpeg;*.png")])
        if caminho_imagem:
            self.img_original = cv2.imread(caminho_imagem)
            self.mostrar_imagem(self.img_original, 'imagem')
    
    def salvar_imagem(self):
        if self.img_processada is not None:
            caminho_salvar = filedialog.asksaveasfilename(defaultextension="*.png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
            if caminho_salvar:
                cv2.imwrite(caminho_salvar, self.img_processada)
                messagebox.showinfo("Sucesso", "Imagem salva com sucesso!")
                
    def mostrar_imagem(self, imagem, titulo):
        #Converte para exibir no Tkinter
        if imagem is not None:
            imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(imagem_rgb)
            imagem_tk = ImageTk.PhotoImage(img_pil)
            self.label_imagem.config(image=imagem_tk)
            self.label_imagem.image = imagem_tk

    def mostrar_histograma(self):
        if self.img_original is not None:
            cinza = p.escala_cinza(self.img_original)
            hist = p.calcular_histograma(cinza)

            plt.plot(hist)
            plt.show()

    def alargamento_contraste(self):
        if self.img_original is not None:
            cinza = p.escala_cinza(self.img_original)
            self.img_processada = p.alargamento_contraste(cinza)

            self.mostrar_imagem(self.img_processada, 'Alargamento')

    def eq_hist(self):
         if self.img_original is not None:
            cinza = p.escala_cinza(self.img_original)
            self.img_processada = p.equalizacao_histograma(cinza)

            self.mostrar_imagem(self.img_processada, 'Alargamento')
        
    def filtro_media (self):
         if self.img_original is not None:
            cinza = p.escala_cinza(self.img_original)
            self.img_processada = p.filtro_media(cinza)

            self.mostrar_imagem(self.img_processada, 'FMedia')
                 
            
