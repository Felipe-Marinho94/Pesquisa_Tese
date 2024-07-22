
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
import kernel 
import pandas as pd
from sklearn.metrics import mean_squared_error


#----------------------------------------------------------------
#Implementação da Proposta 
# Fixed Sized Levenberg-Marquardt LSSVM (FSLM-LSSVM)
#Source:https://www.researchgate.net/publication/377073903_New_Iterative_Pruning_Methods_for_Least_Squares_Support_Vectors_Machines
#----------------------------------------------------------------
#Implementação do Método Fit() para a proposta FSLM-LSSVM

def fit_FSLM_LSSVM(X, y, kappa, mu, Red, N, kernel, tau, epsilon):
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

    #Listas para armazenamento
    idx_suporte = []
    erros = []
    mult_lagrange = []

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
            
            #Armazenando o erro e os índices dos vetores de suporte a cada iteração
            erros.append(np.mean(erro**2))
            idx_suporte.append(-np.abs(np.squeeze(z)).argsort()[:(len(z) - n_colunas_removidas)])

            #Critério de parada
            if np.abs(np.mean(erro_novo**2) - np.mean(erro**2)) < epsilon:
                break
    #

    #Resultados
    resultados = {"Multiplicadores_de_Lagrange": np.squeeze(z),
                  "Erros": erros,
                  "Indices_multiplicadores": idx_suporte}

    #Retornando os multiplicadores de Lagrange finais
    return(resultados)


#------------------------------------------------------------------------------
#Implementação do método predict() para a proposta FSLM_LSSVM 
# considerando um problema de classificação
#------------------------------------------------------------------------------
def predict_class(alphas, b, kernel, X_treino, y_treino, X_teste):
    #Inicialização
    estimado = np.zeros(X_teste.shape[0])
    n_samples_treino = X_treino.shape[0]
    n_samples_teste = X_teste.shape[0]
    K = np.zeros((n_samples_teste, n_samples_treino))
    
    #Construção da matriz de Kernel
    for i in range(n_samples_teste):
        for j in range(n_samples_treino):
            
            if kernel == "linear":
                K[i, j] = linear_kernel(X_teste[i], X_treino[j])
            
            if kernel == "polinomial":
                K[i, j] = polinomial_kernel(X_teste[i], X_treino[j])
            
            if kernel == "gaussiano":
                K[i, j] = gaussiano_kernel(X_teste[i], X_treino[j])
            
        #Realização da predição
        estimado[i] = np.sign(np.sum(np.multiply(np.multiply(alphas, y_treino), K[i])) + b)
    
    return estimado

#Realização de alguns testes
X = np.random.normal(0, 1 , size=(100, 5))
y = np.random.normal(0, 1, 100)
resultados = fit_FSLM_LSSVM(X, y, 2, 0.3, 0.5, 50, "gaussiano", 2, 0.001)
resultados['Erros']
sns.lineplot(resultados, x=range(0,len(resultados['Erros'])), y = resultados['Erros'])
plt.xlabel('Iteração')
plt.ylabel('Erro Médio Quadrático (MSE)')
plt.show()


#Gráfico de dispersão para a Matriz X
sns.set_theme(style="white")
len(resultados['Indices_multiplicadores'])
index = -resultados['Indices_multiplicadores'][0]
index.shape

fig = plt.subplot(2, 2, 1)
fig.scatter(x=X[:,1], y=X[:,2])
fig.plot(X[index,1], X[index,2], "or")

index = -resultados['Indices_multiplicadores'][15]
index.shape
fig1 = plt.subplot(2, 2, 2)
fig1.scatter(x=X[:,1], y=X[:,2])
fig1.plot(X[index,1], X[index,2], "or")

index = -resultados['Indices_multiplicadores'][30]
index.shape
fig2 = plt.subplot(2, 2, 3)
fig2.scatter(x=X[:,1], y=X[:,2])
fig2.plot(X[index,1], X[index,2], "or")

index = -resultados['Indices_multiplicadores'][44]
index.shape
fig3 = plt.subplot(2, 2, 4)
fig3.scatter(x=X[:,1], y=X[:,2])
fig3.plot(X[index,1], X[index,2], "or")

plt.show()

-resultados['Indices_multiplicadores'][0]