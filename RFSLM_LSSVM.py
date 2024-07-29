
"""
Implementação de nova proposta de poda iterativa para
Esparsificação do conjunto de vetores suportes do modelo LSSVM
Data:29/07/2024
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
from sklearn.model_selection import train_test_split

#----------------------------------------------------------------
#Função para o cálculo da matriz A e vetor b para a proposta
#Reduced Set Fixed Sized Levenberg-Marquardt LSSVM (RFSLM-LSSVM)
#----------------------------------------------------------------
def Construct_A_b(X, y, kernel, tau):
    """
    Interface do método
    Ação: Este método ecapsular o cálculo da matriz A e vetor b para
    a proposta RFSLM-LSSVM.

    INPUT:
    X: Matriz de features (array N x p);
    y: Vetor de target (array N x 1);
    kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
    #tau: Parâmetro de regularização do problema primal do LSSVM.

    OUTPUT:
    dois array's representando a matriz A e b, respectivamente.
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
                         np.expand_dims(np.ones(n_samples), axis = 1)), axis=0)
    
    return((A, B))

 
#----------------------------------------------------------------
#Implementação da Proposta 
#Reduced Set Fixed Sized Levenberg-Marquardt LSSVM (RFSLM-LSSVM)
#Source:https://www.researchgate.net/publication/377073903_New_Iterative_Pruning_Methods_for_Least_Squares_Support_Vectors_Machines
#----------------------------------------------------------------
#Implementação do Método Fit() para a proposta FSLM-LSSVM

def fit_RFSLM_LSSVM(X, y, Percent, mu, Red, N, kernel, tau, epsilon):
    """
    Interface do método
    Ação: Este método visa realizar a poda iterativa dos vetores
    de suporte estimados no LSSVM, trabalhando com a matriz dos
    coeficientes reduzida em cada iteração.

    INPUT:
    X: Matriz de features (array N x p);
    y: Vetor de target (array N x 1);
    mu: Taxa de aprendizado (escalar);
    Percent: Porcentagem de redução por iteração de corte (escalar);
    Red: Percentual de remoção (escalar);
    N: Número máximo de iterações (escalar);
    kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
    #tau: iterações de corte (escalar).
    #epsilon: Tolerância (Critério de parada)

    OUTPUT:
    vetor esparso de multiplicadores de Lagrange estimados. 
    """

    #Percentual de treino que representa os vetores de suporte
    Fed = 1 - Red

    #Inicialização aleatória do conjunto com vetores de suporte
    X_rest, X_vs, y_rest, y_vs = train_test_split(X, y, test_size= Fed)

    #Inicialização aleatória do vetor solução do método de Levenberg-Marquardt
    z_inicial = np.random.normal(loc=0, scale=1, size=(X_vs.shape[0] + 1, 1))
    z = z_inicial

    #Listas para armazenamento
    idx_suporte = [range(0, X_vs.shape[0])]
    erros = []
    mult_lagrange = []

    #Loop de iteração
    for k in range(1, N+1):
        
        #Cálculo da matriz A e vetor b para o conjunto de vetores suporte
        A, B = Construct_A_b(X_vs, y_vs, 'gaussiano', 2)

        #Cálculo do erro para a k-ésima iteração
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
        
        if k in np.arange(5, N, 5, dtype='int'):

            #Número de vetores suporte removidos dados pela porcentagem de redução por iteração de corte
            numero_vetores_suporte_removidos = int(Percent * X_vs.shape[0])

            #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
            idx_remover = np.argsort(np.abs(np.squeeze(z)))[:numero_vetores_suporte_removidos-1]

            #Armazenando os índices dos vetores de suporte a cada iteração
            idx_suporte.append(np.argsort(np.abs(np.squeeze(z)))[numero_vetores_suporte_removidos:])

            #Remoção dos vetores suporte de X_vs e multiplicadores de z
            X_vs = np.delete(X_vs, idx_remover, axis = 0)
            z = np.delete(z, idx_remover, axis = 0)

            #Seleção aleatória de Percent dados para serem
            #adicionados ao conjunto de vetores de suporte
            X_add, X_permanece, y_add, y_permanece = train_test_split(X_rest, y_rest, train_size=Percent)
            np.r_(X_vs, [X_add])

            #Inicialização aleatória do vetor z_t P%
            z_percent = np.random.sample(loc=0, scale=1, size=(numero_vetores_suporte_removidos))

            #Atualização dos multiplicadores de Lagrange
            np.r_(z, [z_percent])
        
        #Armazenando o erro
        erros.append(np.mean(erro**2))

        #Critério de parada
        if np.abs(np.mean(erro_novo**2)) < epsilon:
            break

    mult_lagrange = np.zeros(X.shape[0])
    mult_lagrange[idx_suporte[len(idx_suporte)-1]] = np.squeeze(z)[1:len(z)]
    
    #Resultados
    resultados = {"mult_lagrange": mult_lagrange,
                  "b": np.squeeze(z)[0],
                  "Erros": erros,
                  "Indices_multiplicadores": idx_suporte}










