#!/usr/bin/env python
# coding: utf-8

# In[2]:



# coding: utf-8

# In[2]:

import os, glob, random, imageio
import numpy as np
import cv2
import nbimporter


# In[1]:

#recebe um path de diretorios,
#transforma as imagens em lista de pixels
def imgs_para_vetor(path, deteccao=False):
   
    #extension of image
  
    s1 = os.listdir(path+"\\"+os.listdir(path)[0])[0]
   
    ext = (s1[s1.index('.') + len('.'):])
    #print('ext: ', ext)
 
    data = []
    labels = []
    classe = 0
    
    for dir in os.listdir(path):
     
        for image_path in glob.glob(path+"\\"+dir+"\\*."+ext):
            #print('image_path: ', image_path)
            if ext == 'gif':
                ima = imageio.mimread(image_path)
               # print('ima.dtype: ', ima.dtype)
            else:
                ima = cv2.imread(image_path,0)
            
            #print('leu ima')
            #chamar detecção de face
            #input: img de cv2; output: img cv2 com face detectada
            
            #print('ima ', ima)
            ima = np.asarray(ima,dtype="int32" )
            #print(ima.shape)
            h = ima.shape[0]
            w = ima.shape[1]
           # print('h: ', h)
           # print('w: ', w)
            
            ima = ima.flatten()
            
            data.append(ima)
            labels.append(classe)
            
           
        
        classe = classe+1
    
    return data, labels, h, w


# In[3]:

l = []
l.append(2)
l.append(2)
l.append(3)
l.append(3)
l.append(3)
print(l)

i = np.where(np.array(l) == 3)
print(len(i[0])-1)
print(i[0][2])
print(i[0][len(i[0])-1])


# In[4]:

# #separar aleatoriamente treino e teste
def separar_sets(data, labels, perc):
    
    #definindo quantidade de imgs para treino e teste
    n_imgs_tr = int(perc*len(labels))
    n_imgs_te = int(len(labels) - n_imgs_tr)
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    qtd_imgs = len(labels)
    
    set_labels = set(labels)
    n_classes = len(set_labels)
    
    print('n_classe: ', n_classes)
    adicionados = []
    for c in range(n_classes):
        #print('classe: ', c)
        indices = np.where(np.array(labels) == c)
        #print('indices: ', indices)
        qtd_classe = len(indices[0])
        #print('qtd_classe: ', qtd_classe)
        qtd_tr = int(len(indices[0])*perc)
        qtd_te = int(len(indices[0]) - qtd_tr)
        #print('qtd_tr: ', qtd_tr, '; qtd_te: ', qtd_te)
        
        count = 0
        while count < qtd_tr:
            ind = random.randint(indices[0][0], indices[0][qtd_classe-1])
            #print('randomico de ', indices[0][0], ' a ', indices[0][qtd_classe-1],' (incluso)')
            #print('ind-random: ', ind)
            #print('labels[ind]: ', labels[ind])

            x_train.append(data[ind])
            y_train.append(labels[ind])
           
            #remove o vetor de data e labels
            del data[ind]
            del labels[ind]
            
            #print('---depois do remover: ')
            indices = np.where(np.array(labels) == c)
            #print('indices: ', indices)
            #print('\n----------\n')

            qtd_imgs = qtd_imgs - 1
            qtd_classe -= 1
            count += 1
        
        for j in range(int(qtd_te)):
            
            x_test.append(data[0])
            y_test.append(labels[0])
            
            #print('j: ', j, '; labels[0]: ', labels[0])
            del data[0]
            del labels[0]
        
        #print('len(x_train): ', len(x_train))
        #print('len(x_test): ', len(x_test))
   
    
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    
    print('xtrain ', x_train.shape)
    print('ytrain ', y_train.shape)
    print('xtest ', x_test.shape)
    print('ytest ', y_test.shape)
            
    return x_train, y_train, x_test, y_test


# In[5]:

#divide dataset em train e test
def split(path, perc, deteccao=False):
    print('split holdout')
    #transformar imagens do path em vetores de pixels
    data, labels, h, w = imgs_para_vetor(path, deteccao)
    
    #separar aleatoriamente treino e teste
    x_train, y_train, x_test, y_test = separar_sets(data, labels, perc)
    
    return x_train, y_train, x_test, y_test, h, w


# In[ ]:



