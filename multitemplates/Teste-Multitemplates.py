#!/usr/bin/env python
# coding: utf-8

# In[2]:




import nbimporter, os, cv2
import numpy as np
import Imagem
import PixelClustering
import matplotlib.pylab as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.cluster import KMeans


# In[15]:

#pathFormas = "C:\\Users\\ADM\\Desktop\\experimentos"
#pathFormas = "C:\\Users\\Tiago\\Downloads\\imagens\\exp02"
pathFormas = "exp02"

qtd_classes = 2
k = 2
testes = 10

for t in range(testes):
    print("\n---------------Teste ", t," ------------------\n")
    
    x, y, h, w = Imagem.split(pathFormas, qtd_classes)

    f_train, labels_por_classe, qtd_features, y = PixelClustering.intensity_patches_ntemplates(x, y, k, 'variancia', h, w)

    #escrever imagens de cada classe após separação
    indices = np.where(y == 0)

    for i in range(len(indices[0])):
        cv2.imwrite("templates/exp02/0/"+str(t)+"-"+str(i)+".png",np.reshape(x[indices[0][i]],[h,w]))

    indices = np.where(y == 1)

    for i in range(len(indices[0])):
        cv2.imwrite("templates/exp02/1/"+str(t)+"-"+str(i)+".png",np.reshape(x[indices[0][i]],[h,w]))



# In[ ]:




