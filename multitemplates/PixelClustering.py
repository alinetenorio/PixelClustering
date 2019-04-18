#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.cluster import KMeans
import math
import nbimporter
import matplotlib.pylab as plt
import cv2,os

#===================================================

# Recebe um conjunto de imagens e separa em diferentes classes, a partir da semelhança
# entre suas features. As features são extraídas a partir do agrupamento de pixels. 
# É gerado um template para cada classe. De cada grupo de todos templates é extraída uma
# feature. Retorna uma matriz com as features extraídas, cada linha representa uma imagem.
# Retorna uma lista com todos os templates, cada template é um vetor.
# ->x
#   é uma matriz. Cada linha de x corresponde a aos valores de intensidade de
#   todos os pixels de uma imagem. Pode ser uma imagem em tons de cinza ou 
#   colorida. 
# ->y
#   é um vetor, tem o mesmo tamanho das linhas de x. Cada elemento define a 
#   classe de uma das imagens de x.
# <-f
#   matriz onde cada linha consiste em todas as features extraídas para uma imagem
# <-temp_por_classe
#   é uma lista de tamanho igual à quantidade de classes em y. Cada elemento é um template
#   Um template é um vetor de tamanho igual à quantidade de pixels em uma imagem. Cada 
#   elemento do template é o grupo ao qual aquele pixel pertence
# <-qtd_features
#   Inteiro, representa a quantidade de features extraídas por imagem. É igual a 
#   n_cluster * qtd_classes
# <-y
#   vetor de labels após a troca de classes, cada elemento define a classe de uma das imagens de x.
def intensity_patches_ntemplates(x, y, n_cluster,tipo_feature, h=0, w=0, path_template=False):
    #extrair vetores de intensidade
    x = np.array(x)     
    x_list = separar_por_classe(x, y, n_cluster) 
 
    print(y)
    
    y_anterior = y
    
    mudanca = 1
    iteracao = 0    
    while mudanca > 0:
        #mudanca = 0
        iteracao = iteracao + 1
        
        qtd_classes = len(x_list) 
        qtd_features = n_cluster*qtd_classes
    
        temp_por_classe, qtd_por_classe = clusterizar_por_classe(x_list, n_cluster)
        
        f = projetar_ntemplates_editando(x, temp_por_classe, qtd_features, n_cluster, tipo_feature)
        
        print("\nITERACAO "+str(iteracao)+"========================================")
        kmeans = KMeans(n_clusters = n_cluster, init = 'k-means++')
        kmeans.fit(f)   
                
        y = kmeans.labels_
        
        #se o grupo formado nessa iteracao eh igual ao grupo anterior, o algoritmo encerra
        if np.array_equal(y, y_anterior):
            mudanca = 0
        
        y_anterior = y
        
        print("Novos grupos:\n",y)
        x_list = separar_por_classe(x, y, n_cluster)
        print('\n===features===\n', f)
        
        #se o path for passado como argumento, escrever os templates gerados em cada iteracao
        if path_template:
            if not os.path.exists(path_template):
                os.makedirs(path_template)
            for i in range(qtd_classes):
                cv2.imwrite(path_template+"/it"+str(iteracao)+"-temp"+str(i)+".png",255*np.reshape(temp_por_classe[i],[h,w]))
        
  
    return f, temp_por_classe, qtd_features, y


#===================================================

# Cria uma lista de tamanho igual à quantidade de classes
# Cada elemento da lista guarda uma matriz com imagens da 
# mesma classe. Cada linha da matriz representa uma imagem
# ->x
#   é uma matriz. Cada linha de x corresponde a aos valores de intensidade de
#   todos os pixels de uma imagem. Pode ser uma imagem em tons de cinza ou 
#   colorida. 
# ->y
#   é um vetor, tem o mesmo tamanho das linhas de x. Cada elemento define a 
#   classe de uma das imagens de x.
# ->n_cluster
#   é um inteiro, define a quantidade de grupos por template
# <-x_c
#   é uma lista, tem tamanho igual à quantidade de classes diferentes em y.
#   Cada elemento da lista é uma matriz de imagens que pertencem a mesma classe.
#   Cada linha da matriz guarda os valores de intensidade de todos os pixels de 
#   uma imagem.
#   x_c: lista de tamanho igual à quantidade de classes em y
#   x_c[0]: retorna matriz contendo todas as imagens da classe 0.
#   x_c[0][0]: retorna a primeira imagem que pertence à classe 0.
def separar_por_classe(x, y, n_cluster):
    x = np.array(x)
    y = np.array(y)
    
    set_y = set(y) 
   
    #set de treino separado por classes
    x_c = []
   
    for c in set_y:
        ind = np.where(y == c)
    
        for img in range(len(ind)):
            x_c.append(x[ind[img]])
   
    return x_c

#=============================================================
    
# Recebe uma lista, cada elemento da lista é uma matriz de imagens
# que pertencem à mesma classe. Aplica um algoritmo de clusterização
# em cada matriz, gerando um template por classe. Retorna uma lista
# na qual cada elemento é um vetor contendo o template de uma classe
# O vetor de um template tem tamanho igual à quantidade de pixels de 
# uma imagem, ou seja, tamanho igual à len(x[i][j]).
# ->x_list
#   é uma lista, tem tamanho igual à quantidade de classes diferentes em y.
#   Cada elemento da lista é uma matriz de imagens que pertencem a mesma classe.
#   Cada linha da matriz guarda os valores de intensidade de todos os pixels de 
#   uma imagem.
#   x_c: lista de tamanho igual à quantidade de classes em y
#   x_c[0]: retorna matriz contendo todas as imagens da classe 0.
#   x_c[0][0]: retorna a primeira imagem que pertence à classe 0
# ->n_cluster
#   é um inteiro, define a quantidade de grupos por template
# <-template_classe
#   é uma lista. Cada elemento é um vetor. Cada vetor representa um template
#   de uma classe. Cada elemento do vetor corresponde ao grupo em que aquele pixel
#   está contido.
# <-qtd_por_classe
#   é uma lista. Tem tamanho igual à quantidade de classes, ou seja, igual
#   ao tamanho de x_list. Cada elemento da lista guarda um inteiro que representa
#   a quantidade de imagens em uma classe
#clusteriza e retorna o vetor labels_ de cada cluster-classe
def clusterizar_por_classe(x_list, n_cluster):
  
    template_classe = []
    qtd_por_classe = []
    kmeans = KMeans(n_clusters = n_cluster, init = 'k-means++')
    for x_classe in x_list:
        x_c = (np.array(x_classe)).transpose()
        kmeans.fit(x_c)
        
        template_classe.append(kmeans.labels_)
        qtd_por_classe.append(len(x_c[0])) 
    
    return template_classe, qtd_por_classe


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
# tipo_feature
#   é o tipo de feature extraída, pode ser média ou variância de cada grupo.
def projetar_ntemplates_editando(x, labels_por_classe, qtd_features, n_cluster, tipo_feature):
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
                
            f[i, inicio_feat:fim_feat] = extrair_feat(x[i, :], labels_por_classe[n], n_cluster, tipo_feature)

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
# tipo_feature
#   é o tipo de feature a ser extraída.
def extrair_feat(x, labels, n_cluster, tipo_feature):
    
    # f é a matriz de respostas com as features extraídas. Cada linha
    #   corresponde as features da imagem da linha equivalente em x.
    f = np.zeros(shape=(n_cluster))
        
    # faz a extração de features para cada imagem
   
    for icluster in range(n_cluster): 

        #todos os indices de vetores que estao contido no mesmo cluster
        #cada elemento corresponde à posição do vetor de x que contém
        #um pixel pertencente a icluster
        indices = np.where(labels == icluster)

        feature = percorrer_cluster(x, indices, tipo_feature)

        f[icluster] = feature
            
    return f

#=================================================================

# Percorre todos os pixels que pertencem ao mesmo cluster de um template.
# Extrai uma feature do cluster. Pode ser a média da intensidade de todos
# os pixels que pertencem ao cluster. Ou pode ser a variância dos pixels
# do cluster dividida pela variância total da imagem. Isso é feito para que
# os valores de intensidade dos pixels não interfira na extração da feature.
# ->x
#  é um vetor numérico de features bruto. Do qual serão extraídas novas features.
# ->indices
#  é um vetor de inteiros, possui todos os indices dos elementos de x que 
#  pertencem a um mesmo cluster
# ->tipo_feature
#  define o tipo de feature a ser extraída. Pode ser média ou variância
# <-feature
#  valor numérico da nova feature extraída
def percorrer_cluster(x, indices, tipo_feature):
    #percorrer cada vetor do mesmo cluster, selecionar pxl de uma img
    soma = 0
    media = 0
            
    for pxl in indices[0]:
        soma = soma + x[pxl]
            
    media = soma / len(indices[0])
    
    if tipo_feature == 'media':
        return media
    
    elif tipo_feature == 'variancia':
     
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
      
        return feature
    
#===========================================================

#Recebe um vetor de valores numéricos e retorna a variância dos dados
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




