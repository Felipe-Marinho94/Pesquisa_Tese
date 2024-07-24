"""
Implementação das propostas para a tese
Data:02/04/2024
"""

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
import math
from numpy import linalg
from Gradiente_conjugado import ichol
from ADMM import ADMM

#------------------------------------------------------------------------------
#Implementação de algumas funções relevantes
#------------------------------------------------------------------------------
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def linear_kernel(x, x_k):
    return np.dot(x, x_k)

def polinomial_kernel(x, y, C = 1, d = 3):
    #Inputs
    #x: vetor x
    #y: vetor y
    #C: Constante
    #d: Grau do Polinômio
    
    return (np.dot(x, y) + C)**d

def gaussiano_kernel(x, y, gamma = 1):
    return np.exp(-gamma * linalg.norm(x - y)**2)


#------------------------------------------------------------------------------
#Implementação do método Fit() para a primeira proposta
#baseada na esparsificação do LSSVM utilizando regularização L1 (LASSO)
#no problema primal e resolução do problema de otimização pela aplicação
#do algoritmo Alternating Directions Multipliers Method (ADMM)
##Solução aproximada de problemas do tipo min ||b - Ax||^(2) + lambda*||alpha||_{1}
#------------------------------------------------------------------------------
def fit_LSSVM_ADMM(X, y, tau, kernel):
    #Input
    #X: Matriz de Dados (array n x p)
    #y: Vetor de rótulos (classificação) 
    #   ou vetor de respostas numéricas (regressão) (array n x 1)
    #kernel: kernel utlizado ("linear", "polinomial", "gaussiano") (string)
    #tau: Termo de regularização do problema primal do LSSVM (escalar)
    #Output
    #x_ótimo: Vetor solução aproximado do sistema KKT Ax = b (array n+1 x 1)
    #x_ótimo = [alphas b].T
    #b = 0
    
    
    #--------------------------------------------------------------------------
    #Obtenção da matriz de kernel
    #--------------------------------------------------------------------------
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
    
    #--------------------------------------------------------------------------
    #Decomposição da matriz de kernel
    #--------------------------------------------------------------------------
    #Cholesky Incompleta
    P = np.linalg.cholesky(K + 0.01 * np.diag(np.full(K.shape[0], 1)))
    
    #Construção da matriz dos coeficiente A
    A = P.T
    
    #Construção do vetor de coeficientes b
    b = np.dot(linalg.inv(tau * np.identity(n_samples) + np.dot(P, P.T)), np.dot(P.T, y))
    b = np.expand_dims(b, axis = 1)
    
    #Solução do sistema KKT em conjunto com o LASSO
    solution, rho, l  = ADMM(A, b)
    
    #Obtenção dos multiplicadores de Lagrange
    alphas = solution
    
    resultado = {"mult_lagrange": alphas,
                 "kernel": K,
                 "A": A,
                 "b": 0}
    
    return resultado

#------------------------------------------------------------------------------
#Implementação do método predict() para a primeira proposta considerando um
#problema de classificação
#------------------------------------------------------------------------------
def predict_class_LSSVM_ADMM(alphas, b, kernel, X_treino, y_treino, X_teste):
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
        estimado[i] = np.sign(np.sum(np.multiply(alphas, K[i])) + b)
    
    return estimado

