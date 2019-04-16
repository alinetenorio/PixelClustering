#!/usr/bin/env python
# coding: utf-8

# In[21]:



# coding: utf-8

# In[1]:

import numpy as np
from sklearn.cluster import KMeans
import math
import nbimporter
import matplotlib.pylab as plt
import cv2

#===================================================

def intensity_patches_ntemplates(x_train, y_train, n_cluster,projecao, h=0, w=0):
    #extrair vetores de intensidade
    x_train = np.array(x_train)     
    x_list = separar_por_classe(x_train, y_train, n_cluster) 
    
    print("1========================================")
    print(y_train)
    
    y_train_ant = y_train
    
    mudanca = 1
    interacao = 0    
    while mudanca > 0:
        #mudanca = 0
        interacao = interacao + 1
        
        qtd_classes = len(x_list) 
        qtd_features = n_cluster*qtd_classes
    
        labels_por_classe, qtd_por_classe = clusterizar_por_classe(x_list, n_cluster)
        
        f_train = projetar_ntemplates_editando(x_train, labels_por_classe, qtd_features, n_cluster, projecao)
        
        print("ITERACAO "+str(interacao)+"========================================")
        kmeans = KMeans(n_clusters = n_cluster, init = 'k-means++')
        kmeans.fit(f_train)   
                
        y_train = kmeans.labels_
        
        #se o grupo formado nessa iteracao eh igual ao grupo anterior, o algoritmo encerra
        if np.array_equal(y_train, y_train_ant):
            mudanca = 0
        
        y_train_ant = y_train
        
        print("Novos grupos:\n",y_train)
        x_list = separar_por_classe(x_train, y_train, n_cluster)
        print("qtd_por_classe[0]: ",qtd_por_classe[0])
        print("qtd_por_classe[1]: ",qtd_por_classe[1])
        print('len(x_list[0]): ', len(x_list[0]))
        #print('len(x_list[1]): ', len(x_list[1]))
        
        print('\n===ftrain===\n', f_train)
        for i in range(qtd_classes):
            print('escrevendo imagem ')
            print('path: ', "templates/temp"+str(i)+".png")
            cv2.imwrite("templates/temp"+str(interacao)+"-"+str(i)+".png",255*np.reshape(labels_por_classe[i],[h,w]))
        
   
    
   # f_train = projetar_ntemplates(x_train, labels_por_classe, qtd_features, qtd_por_classe, qtd_classes, n_cluster, projecao)
    
    
    return f_train, labels_por_classe, qtd_features, y_train


#===================================================

#TODO: ALTERAR modo de separar as classes, utilizar indices
def separar_por_classe(x_train, y_train, n_cluster):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    set_y_train = set(y_train) 
   
    #set de treino separado por classes
    x_train_c = []
   
    for c in set_y_train:
        ind = np.where(y_train == c)
        print("---->indices por classe: "+ str(c)+": ", ind)
       
        for img in range(len(ind)):
            x_train_c.append(x_train[ind[img]])
    
    #print(x_train_c)
   
    return x_train_c

#=============================================================
    
#recebe uma lista com elementos divididos por classe
#clusteriza e retorna o vetor labels_ de cada cluster-classe
def clusterizar_por_classe(x_list, n_cluster):
    print('len(x_list)',len(x_list))
    print('len(x_list[0])',len(x_list[0]))
    print('len(x_list[0][0])',len(x_list[0][0]))
    print('len(x_list[0][1])',len(x_list[0][1]))
    print('len(x_list[0][2])',len(x_list[0][2]))
    print('len(x_list[0][3])',len(x_list[0][3]))
    #print('len(x_list[0][4])',len(x_list[0][4]))
    print('len(x_list[1])',len(x_list[1]))
    y_por_classe = []
    qtd_por_classe = []
    kmeans = KMeans(n_clusters = n_cluster, init = 'k-means++')
    for x_classe in x_list:
        x_c = (np.array(x_classe)).transpose()
        kmeans.fit(x_c)
        
        y_por_classe.append(kmeans.labels_)
        qtd_por_classe.append(len(x_c[0])) 
    
    print('len(y_por_classe): ',len(y_por_classe))
    print('len(y_por_classe[0]): ',len(y_por_classe[0]))
    return y_por_classe, qtd_por_classe


#=============================================================

# x 
#   é uma matriz. Cada linha de x corresponde a aos valores de intensidade de
#   todos os pixels de uma imagem. Pode ser uma imagem em tons de cinza ou 
#   colorida. Cada linha corresponte à concatenação das linhas ou colunas da
#   matriz que representa a imagem.
# labels_por_classe 
#   é uma matriz. Matriz de templates. Cada linha desta matriz tem o mesmo número de
#   dimensões que as imagens de x, ou seja, tem o mesmo número de colunas que x.
#   Cada valor em uma linha é o índice de um grupo, este índice indica a qual
#   grupo pertence o pixel da posição correspondentes.
# qtd_features
#   é a quantidade classes vezes a quantidade de grupos em cada template.
#   A quantidade de classes também pode ser calculada como o número de linhas
#   em labels_por_classe.
# n_cluster
#   é quantidade de grupos em cada template.
# projecao
#   é o tipo de feture extraída, pode ser média ou variância de cada grupo.
def projetar_ntemplates_editando(x, labels_por_classe, qtd_features, n_cluster, projecao):
    # inicializa com zero o vetor de características as ser extraído.
    f = np.zeros( shape=(len(x), qtd_features ) )
       
    # inicio_feat é
    # posicao inicial das fetures para uma imagem
    # é incremenatado para concatenar as features
    # de um novo template logo após inserir as 
    # features do template anterior
    inicio_feat = 0 
    
    # quantidade de templates
    qtd_classes = int(qtd_features/n_cluster)
    
    for i in range(len(x)): # varre cada imagem
        for n in range(qtd_classes): # para cada template
            # separar features por template
            
            # fim_feat é final + 1 das features extraídas
            # para o template n na image i.
            fim_feat = n*n_cluster + n_cluster           
                
            f[i, inicio_feat:fim_feat] = extrair_feat(x[i, :], labels_por_classe[n], n_cluster, projecao)

            inicio_feat = fim_feat

        #aqui temos todas as variancias de todos os grupos e todos os templates
        inicio_feat = 0 
        
    return f  # vetor de características as ser extraído.

#============================================================

# extrair_feat(x, labels, n_cluster, projecao)
#   extraí fetures de x conforme o template labels (com n_cluster
#   diferentes grupos). Retorna um vetor de fetures extraídas se x for um vetor.
#   ou retorna uma matriz de fetures extraídas se x for uma matriz e cada
#   linha de x for um vetor de features.
# x
#   é um vetor numérico de features bruto. Do qual serão extraídas novas features.
# lables
#   é um template. Representado por um vetor de inteiros onde cada valor é o
#   índice do grupo daquele template. Os grupos são indexados de 0 a n_cluster-1.
# n_cluster
#   é número de diferentes grupos no template
# projecao
#   é o tipo de feature a ser extraída.
def extrair_feat(x, labels, n_cluster, projecao):
    print("labels: ", labels)
    
    # f é a matriz de respostas com as features extraídas. Cada linha
    #   corresponde as features da imagem da linha equivalente em x.
    f = np.zeros(shape=(n_cluster))
        
    # faz a extração de fetaures para cada imagem
   
    for icluster in range(n_cluster): 

        #todos os indices de vetores que estao contido no mesmo cluster
        indices = np.where(labels == icluster)

        feature = percorrer_cluster(x, indices, projecao)

        f[icluster] = feature
            
    return f

#=================================================================

def percorrer_cluster(x, indices, projecao):
    #percorrer cada vetor do mesmo cluster, selecionar pxl de uma img
    soma = 0
    media = 0
            
    for pxl in indices[0]:
        soma = soma + x[pxl]
            
    media = soma / len(indices[0])
    
    if projecao == 'media':
        return media
    
    elif projecao == 'variancia':
     
        diferenca = 0
        for pxl in indices[0]:
            
            #print('x-',pxl,': ', x[pxl])
            #print('x[pxl] - media: ', x[pxl] - media)
            #print('(x[pxl] - media)**2 ', (x[pxl] - media)**2 )
            diferenca += (x[pxl] - media)**2 
            #print('dif: ', diferenca)
         
        #print('len(indices[0]): ', len(indices[0]))
        #variancia do agrupamento de features, representado pelo vetor de indices
        variancia = diferenca/len(indices[0])
        
        #variancia total do vetor x
        variancia_x = variancia_total(x)
        
        feature = variancia/variancia_x
        print('variancia cluster: ', variancia)
        print('variancia total: ', variancia_x)
        print('feature: ', feature)
        return feature
    
#===========================================================

def variancia_total(x):
    soma = 0
    media = 0
            
    for i in range(len(x)):
        soma = soma + x[i]
                
    media = soma / len(x)
    
    diferenca = 0
    for i in range(len(x)):
        diferenca += (x[i] - media)**2 

    variancia = diferenca/len(x)
    
    return variancia


# In[ ]:




