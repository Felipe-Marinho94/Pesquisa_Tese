'''
Implemetação do método Conjugate Functional Gain 
Sequential Minimal Optimization (CFGSMO) para treinamento
do modelo de máquinas de vetores suporte de mínimos quadrados (LSSVM)
Data:12/08/2024 
'''

#---------------------------------------------------------------
#Carregando alguns pacotes relevantes
#---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kernel import gaussiano_kernel, linear_kernel, polinomial_kernel

#---------------------------------------------------------------
#Função para ajustar o modelo LSSVM utilizando o método CFGSMO
#SOURCE: https://link.springer.com/article/10.1007/s00521-022-07875-1
#---------------------------------------------------------------
def fit_CFGSMO_LSSVM(X, y, gamma, kernel, epsilon):
    '''
    Interface do método
    Ação: Este método visa realizar o ajuste do modelo LSSVM empregando
      a metodologia CFGSMO, retornando os multiplicadores de lagrange
      ótimos para o problema de otimização.

    INPUT:
    X: Matriz de features (array N x p);
    y: Vetor de target (array N x 1);
    gamma: termo de regularização de Tichonov (escalar)
    kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
    epsilon: Tolerância (Critério de parada)

    OUTPUT:
    vetor ótimo de multiplicadores de Lagrange estimados.
    '''
    #Inicialização
    #Multiplicadores de Lagrange
    alphas = np.zeros(X.shape[0], 1)

    #Gradiente
    Gradient = np.zeros(X.shape[0], 1)

    #direções conjugadas
    s = np.zeros(X.shape[0], 1)
    t = np.zeros(X.shape[0], 1)

    #Termo tau
    tau = 1

    #Construindo a matriz de kernel
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
            
    
    #Regularização de Tichonov
    K_tilde = K + (1/gamma)*np.diag(np.full(K.shape[0], 1))

