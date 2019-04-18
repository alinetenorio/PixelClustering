#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nbimporter, os, cv2
import numpy as np
import Imagem
import PixelClustering
import matplotlib.pylab as plt

# Função utilizada para escrever todas as imagens separadas por classe
def plotar_classes(path, y, h, w, t=0):
    
    qtd_classes = len(set(y))
    
    for classe in range(qtd_classes):
        indices = np.where(y == classe)

        for i in range(len(indices[0])):
            path_classes = 'classes/'+path+"/"+str(classe)
            if not os.path.exists(path_classes):
                os.makedirs(path_classes)
    
            cv2.imwrite(path_classes+"/teste"+str(t)+str(i)+".png",np.reshape(x[indices[0][i]],[h,w]))
    
#=================================================================

#->Datasets utilizados nos experimentos
#quadrados e circulos: dataset autoral
pathFormas = "exp01" 
#devices: classes do MPEG7
pathDevices = "exp02"

path = pathFormas

# Quantidade de classes em que queremos separar o dataset
qtd_templates = 2
# Quantidade de grupos por template
k = 2
# Quantidade de testes, cada um iniciando com uma partição aleatória das imagens
testes = 1

for t in range(testes):
    print("\n---------------Teste ", t," ------------------\n")
    
    x, y, h, w = Imagem.split(path, qtd_templates)

    f, templates_classe, qtd_features, y = PixelClustering.intensity_patches_ntemplates(x, y, k, 'variancia', h, w, "templates/"+path)

    #escrever imagens de cada classe após separação
    plotar_classes(path, y, h, w, t)

