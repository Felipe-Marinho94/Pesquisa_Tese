'''
Implementação de metodologia heurística para obtenção de uma aproximação de posto reduzido
para a matriz de kernel no método de máquinas de vetores de suport por mínimos quaddrados (LSSVM)
Data:24/09/2024
'''

#-------------------------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#-------------------------------------------------------------------------------------------------
import numpy as np
import random
from scipy import linalg as ln
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from kernel import linear_kernel, polinomial_kernel, gaussiano_kernel

#-------------------------------------------------------------------------------------------------
#Implementando algumas funções relevantes
#-------------------------------------------------------------------------------------------------
def HAN_BASIC(K, b, r, N):
    '''
    Implementação de metodologia heurística para obtenção de uma aproximação de posto reduzido
    para a matriz de kernel no método de máquinas de vetores de suporte por mínimos quadrados (LSSVM).
    Tal aproximação utiliza decomposições QR com pivotamento para um esquema de refinamento sucessivo.
    -INPUT
    K - array representando a matriz de kernel: array de dimensão (n x n);
    b - Tamanho do passo de amostragem: inteiro;
    r - posto numérico desejado para aproximação: inteiro;
    N - Número máximo de iterações. 
    '''

    #Inicialização do conjunto de índices para as colunas pivotadas
    J = [] #Vazio
    columns_index = range(K.shape[1])
    I = range(K.shape[0])
    
    #Inicilaização do termo de iteração
    k = 0

    #loop principal
    while k <= N:
        
        #Seleção aleatória inicial do conjunto de índices para as colunas
        J_hat = random.sample(set(columns_index).difference(set(J)), b)
        
        #Obtenção do conjunto de índices J_hat 
        J_tilde = set(J).union(set(J_hat))
        J_tilde = np.array(list(J_tilde))
        
        #Obtenção do conjunto de índices de linhas I_bar
        I_bar = I
        
        #Etapa de pivotamento de linhas utilizando as colunas amostradas e a
        #decomposição QR com pivotamento
        Q, R, I = ln.qr(K.T[:, J_tilde], pivoting = True)
        
        #Atualização do conjunto de índices I_hat
        I_hat = set(I).difference(set(I_bar))
        I_hat = np.array(list(I_hat))
        
        #Etapa de pivotamento de colunas utilizando as linhas obtidas na etapa
        #anterior em conjunto com a decomposição QR com pivotamento
        Q, R, J = ln.qr(K[I, :].T, pivoting = True)
        
        #Incremento
        k = k + 1

        #Condição de parada
        if set(I_hat) == set():
            break
        
        if (len(I) == r or len(J) == r):
            break
    
    #Resultados
    if len(I) <= len(J):
        J = I
    else:
        I = J
    
    K_reduzido = []
    for i in I:
        for j in J:
            K_reduzido.append(K[i, j])
    
    K_reduzido = np.array(K_reduzido)
    K_reduzido = K_reduzido.reshape((len(I), len(J)))


    resultados = {'I': I,
                  'J': J,
                  'K_reduzido': K_reduzido}
    
    return(resultados)

        
#-------------------------------------------------------------------------------------------------
#Realizando alguns testes
#-------------------------------------------------------------------------------------------------
K = np.random.normal(loc = 0, scale = 1, size = (100, 100))
J = [] #Vazio
columns_index = range(K.shape[1])
I = range(K.shape[0])
    
#Inicilaização do termo de iteração
k = 0

#loop principal
while k <= 40:
        
        #Seleção aleatória inicial do conjunto de índices para as colunas
        J_hat = random.sample(set(columns_index).difference(set(J)), 10)
        
        #Obtenção do conjunto de índices J_hat 
        J_tilde = set(J).union(set(J_hat))
        J_tilde = np.array(list(J_tilde))
        
        #Obtenção do conjunto de índices de linhas I_bar
        I_bar = I
        
        #Etapa de pivotamento de linhas utilizando as colunas amostradas e a
        #decomposição QR com pivotamento
        Q, R, I = ln.qr(K.T[:, J_tilde], pivoting = True)
        
        #Atualização do conjunto de índices I_hat
        I_hat = set(I).difference(set(I_bar))
        I_hat = np.array(list(I_hat))
        
        #Etapa de pivotamento de colunas utilizando as linhas obtidas na etapa
        #anterior em conjunto com a decomposição QR com pivotamento
        Q, R, J = ln.qr(K[I, :].T, pivoting = True)
        
        #incremento
        k = k + 1
        
        #Condição de parada
        if set(I_hat) == set():
            break
        
        if (len(I) == 70 or len(J) == 70):
            break
J_tilde
J_hat
I_bar
len(I)
I_hat
len(J)
I
J
K[[1, 3, 4], J]
resultados = HAN_BASIC(K, 10, 70, 10)
resultados['K_reduzido'].shape
resultados['I'].shape
resultados['J'].shape
I = resultados['I']
J = resultados['J']
K_reduzido = []
for i in I:
    for j in J:
        K_reduzido.append(K[i, j])

K_reduzido = np.array(K_reduzido)
print(K_reduzido)
K_reduzido = K_reduzido.reshape((10, 10))
K_reduzido.shape