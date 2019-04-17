#!/usr/bin/env python
# coding: utf-8

# In[41]:


import os, glob, random, imageio
import numpy as np
import cv2
import nbimporter

#====================================================

# Recebe o path para um diretório contendo dados sem rótulo e
# em quantas classes esses dados serão divididos. Associa uma classe
# aleatoriamente para cada amostra. A quantidade de amostras em uma
# classe é definida pela quantidade de imagens dividida pela quantidade
# de classes. Retorna uma matriz com os dados, onde cada linha da matriz
# é um vetor; o vetor de labels e as dimensões originais dos dados (altura
# e largura se o dado lido for uma imagem)
# ->path:
#    Caminho para o diretório que contém um conjunto de imagens
#    sem rótulo
# ->qtd_classes:
#    Indica em quantas classes as imagens devem ser divididas
# <-data
#    Lista onde cada elemento é um vetor com as features brutas
#    de uma imagem
# <-y_classe
#    Vetor com as labels geradas. Cada elemento i do vetor representa a
#    classe do elemento i da lista de data.Tem tamanho igual à quantidade
#    de elementos de data.
# <-h e w
#    h: altura original da imagem; w: largura original da imagem
def split(path, qtd_classes):
    #transformar imagens do path em vetores de pixels
    data, h, w = imgs_para_vetor(path)
    
    #separar aleatoriamente treino e teste
    y_classe = separar_sets(len(data), qtd_classes)
    
    return data, y_classe, h, w

#===================================================

# Recebe o path para um diretório contendo imagens, lê as imagens
# e transforma cada uma em um vetor de dimensão altura*largura.
# Cada vetor gerado é adicionado à uma lista. Retorna a lista de 
# vetores e as dimensões originais da imagem (altura e largura)
# ->path
#     Caminho para um diretório que contém imagens
# <-data
#     Lista de vetores, onde cada elemento é um vetor com as features brutas
#     de uma imagem
# <-h e w
#     h: altura original da imagem; w: largura original da imagem
#transforma as imagens em lista de pixels
def imgs_para_vetor(path):
   
    #extensão do dado
    i = glob.glob(path+"\\*")[0]
    ext = (i[i.index('.') + len('.'):])
 
    data = []
    for image_path in glob.glob(path+"\\*."+ext):
        #opencv não tem suporte para imagens .gif
        if ext == 'gif':
            ima = imageio.mimread(image_path)
        else:
            #parâmetro 0: lê a imagem como greyscale
            ima = cv2.imread(image_path,0)

        ima = np.asarray(ima,dtype="int32" )

        h = ima.shape[0]
        w = ima.shape[1]
       
        ima = ima.flatten()

        data.append(ima)
        
    return data, h, w

#=======================================================

# Dadas a quantidade de amostras e a quantidade de classes desejadas,
# associa, aleatoriamente, cada amostra à uma classe. A diferença entre
# a quantidade de amostras de uma classe para outra não é maior que 1.
# ->qtd_data
#   Quantidade de amostras
# ->qtd_classes
#   Quantidade de classes que irão ser criadas
# <-y_classe
#   Vetor com as labels geradas. Cada elemento i do vetor representa a
#   classe da amostra i.Tem tamanho igual à quantidade de amostras. 
def separar_sets(qtd_data, qtd_classes):
    
    n_g = np.zeros(shape=qtd_classes)
    
    #Definir a quantidade de elementos por classe: divide a quantidade de amostras 
    #pela quantidade de classes
    q = int(qtd_data/qtd_classes)
    r = qtd_data%qtd_classes
    
    # se o resto for igual a zero, todas as classes terão a mesma quantidade de amostras
    # se for diferente, garantimos que a diferenca de amostras de uma classe para a outra
    # é a menor possível. Cada classe tem no mínimo q elementos e, se r > 0, as primeiras
    # r classes irão ter q+1 elementos
    for i in range(qtd_classes):
        n_g[i] += q
        if r > 0:
            n_g[i] += 1
            r -= 1
            
    y_classe = np.zeros(shape=qtd_data)
    
    # Para cada classe, números aleatórios são gerados para definir
    # o elemento que vai pertencer aquela classe.
    adicionados = []
    for classe in range(qtd_classes):
        
        count = 0
        while count < n_g[classe]:
            ind = random.randint(0, qtd_data-1)
            
            if ind not in adicionados:
                y_classe[ind] = classe
                adicionados.append(ind)
                count += 1
            
            
    return y_classe

