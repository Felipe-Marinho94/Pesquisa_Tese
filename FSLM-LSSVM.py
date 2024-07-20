"""
Implementação das duas novas propostas de poda iterativa para
Esparsificação do conjunto de vetores suportes do modelo LSSVM
Data:15/05/2024
"""

#----------------------------------------------------------------
#Carregando alguns pacotes
#----------------------------------------------------------------
import numpy as np
from numpy import linalg
import math as mt
import matplotlib.pyplot as plt
import seaborn as sns
from kernel import linear_kernel, polinomial_kernel, gaussiano_kernel
import pandas as pd
from sklearn.metrics import mean_squared_error


#----------------------------------------------------------------
#Implementação das Propostas
#----------------------------------------------------------------
#Fixed Sized Levenberg-Marquardt LSSVM (FSLM-LSSVM)
def FSLM_LSSVM(X, y, kappa, mu, Red, N, kernel, tau, epsilon):
    """
    Interface do método
    Ação: Este método visa realizar a poda iterativa dos vetores
    de suporte estimados no LSSVM, trabalhando com a matriz dos
    coeficientes completa em todas as iterações.

    INPUT:
    X: Matriz de features (array N x p);
    y: Vetor de target (array N x 1);
    mu: Taxa de aprendizado (escalar);
    kappa: Termo que define a faixa de poda (escalar);
    Red: Percentual de remoção (escalar);
    N: Número máximo de iterações (escalar);
    kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
    #tau: termo de regularização do problema primal do LSSVM (escalar).
    #epsilon: Tolerância (Critério de parada)

    OUTPUT:
    vetor esparso de multiplicadores de Lagrange estimados. 
    """

    #Construção da matriz A e vetor b e inicialização aleatória
    #do vetor de multiplicadores de Lagrange
    
    n_samples, n_features = X.shape
    
    #Matriz de Gram
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            
            #Kernel trick
            if kernel == "linear":
                K[i, j] = linear_kernel(X[i], X[j])
            
            if kernel == "gaussiano":
                K[i, j] = gaussiano_kernel(X[i], X[j])
            
            if kernel == "polinomial":
                K[i, j] = polinomial_kernel(X[i], X[j])
    
    #Construção da Matriz Omega
    Omega = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            Omega[i, j] = y[i] * y[j] * K[i, j]
    
    #Construção da Matriz A
    H = Omega + (1/tau) * np.identity(n_samples)
    A = np.block([[np.array([0]), np.expand_dims(y, axis = 1).T],
                   [np.expand_dims(y, axis = 1), H]])

    #Construção do Vetor B
    B = np.concatenate((np.expand_dims(np.zeros([1]), axis=1),
                         np.expand_dims(np.ones(100), axis = 1)), axis=0)
    
    #Inicialização aleatória do vetor de multiplicadores de Lagrange
    z_inicial = np.random.normal(loc=0, scale=1, size=(n_samples + 1, 1))
    z = z_inicial

    #Loop de iteração
    for k in range(0, N + 1):

        #Calculando o erro associado
        erro = B - np.matmul(A, z)

        #Atualização
        z_anterior = z
        z = z + np.matmul(linalg.inv(np.matmul(A.T, A) + 0.1 * np.diag(np.matmul(A.T, A))), np.matmul(A.T, erro))

        #Condição para manter a atualização
        erro_novo = B - np.matmul(A, z)
        if np.mean(erro_novo**2) < np.mean(erro**2):
            z = z
        else:
            mu = mu/10
            z = z_anterior
        
        #Condição para a janela de poda
        if k > kappa and k < N - kappa:
            
            #Realização da poda
            n_colunas_removidas = int((n_samples - n_samples * Red)/(N - 2 * kappa))

            #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
            idx_remover = np.abs(np.squeeze(z)).argsort()[:n_colunas_removidas]

            #Remoção das colunas de A e linhas de z
            A = np.delete(A, idx_remover, axis = 1)
            z = np.delete(z, idx_remover, axis = 0)

            #Outra condição
            if k == N- kappa:

                #Realização da poda
                n_colunas_removidas = int((n_samples - n_samples * Red)%(N - 2 * kappa))

                #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
                idx_remover = np.abs(np.squeeze(z)).argsort()[:n_colunas_removidas]

                #Remoção das colunas de A e linhas de z
                A = np.delete(A, idx_remover, axis = 1)
                z = np.delete(z, idx_remover, axis = 0)
            
            #Critério de parada
            if np.abs(np.mean(erro_novo**2) - np.mean(erro**2)) < epsilon:
                break

    #Retornando os multiplicadores de Lagrange finais
    return(np.squeeze(z))



            
#Realização de alguns testes
d1 = np.zeros([1])
d1 = np.expand_dims(d1, axis = 1)
d2 = np.expand_dims(np.ones(100), axis = 1)
d1.shape
d2.shape
B = np.concatenate((d1, d2), axis = 0)
#B = np.squeeze(B)
B = np.concatenate((np.expand_dims(np.zeros([1]), axis=1),
                         np.expand_dims(np.ones(100), axis = 1)), axis=0)
B

z_inicial = np.random.normal(loc=0, scale=1, size=(100 + 1, 1))
A = np.random.normal(0, 1, size=(101, 101))
A.shape

erro = B - np.matmul(A, z_inicial) 
np.mean(erro**2)
np.matmul(np.linalg.inv(np.matmul(A.T, A) + 0.3 * np.diag(np.matmul(A.T, A))), np.matmul(A.T, erro))

idx_remover = np.squeeze(np.abs(z_inicial)).argsort()[:10]
idx_remover
np.delete(A, idx_remover, axis=1).shape
np.delete(z_inicial, idx_remover, axis=0).shape
X = np.random.normal(0, 1 , size=(100, 5))
y = np.random.normal(0, 1, 100)
FSLM_LSSVM(X, y, 2, 0.3, 0.5, 50, "gaussiano", 2, 0.001).shape
