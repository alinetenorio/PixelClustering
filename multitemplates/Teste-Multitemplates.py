#!/usr/bin/env python
# coding: utf-8

# In[1]:




import nbimporter, os, cv2
import numpy as np
import Holdout
import PixelClustering
import matplotlib.pylab as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.cluster import KMeans


# In[15]:

#pathFormas = "C:\\Users\\ADM\\Desktop\\experimentos"
#pathFormas = "C:\\Users\\Tiago\\Downloads\\imagens\\exp02"
pathFormas = "exp02"

x, y, h, w = Holdout.imgs_para_vetor(pathFormas)

xx = x
k = 2

f_train, labels_por_classe, qtd_features, y = PixelClustering.intensity_patches_ntemplates(x, y, k, 'variancia', h, w)

#escrever imagens separadas

indices = np.where(y == 0)
print('indices: ',indices[0])

for i in range(len(indices[0])):
    print('for')
    cv2.imwrite("templates/exp02/0/"+str(i)+".png",np.reshape(x[indices[0][i]],[h,w]))
    
indices = np.where(y == 1)
print('indices: ',indices)

for i in range(len(indices[0])):
    print('for')
    cv2.imwrite("templates/exp02/1/"+str(i)+".png",np.reshape(x[indices[0][i]],[h,w]))
   

print("f_train: ", f_train)



# In[ ]:




