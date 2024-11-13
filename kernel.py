"""
Implementação dos métodos fit() e predict() para o algoritmo
LSSVM utilzando método do Gradiente Conjugado de Hestens-Stiefel
Data:13/04/2024
"""

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import linalg
from sklearn.datasets import make_blobs, make_regression
from matplotlib import style
from Gradiente_conjugado import CG, CG_conditioned

#------------------------------------------------------------------------------
#Implementação de algumas funções de kernel
#Linear, Polinomial, Gaussiano
#------------------------------------------------------------------------------
def linear_kernel(x, x_k):
    return np.dot(x, x_k)

def polinomial_kernel(x, y, C = 1, d = 3):
    #Inputs
    #x: vetor x
    #y: vetor y
    #C: Constante
    #d: Grau do Polinômio
    
    return (np.dot(x, y) + C)**d

def gaussiano_kernel(x, y, gamma = 0.5):
    return np.exp(-gamma * linalg.norm(x - y)**2)


#------------------------------------------------------------------------------
#Implementação do método fit() para problemas de classificação e regressão
#utilizando o CG de Hestenes-Stiefel
#fontes: 
#    -SUYKENS, Johan AK; VANDEWALLE, Joos. Least squares support vector machine classifiers.
#     Neural processing letters, v. 9, p. 293-300, 1999.
#
#    -GOLUB, Gene H.; VAN LOAN, Charles F. Matrix computations. JHU press, 2013.   
#------------------------------------------------------------------------------
def fit_LSSVM(X, y, tau, kernel):
    #Inputs
    #X: array das variáveis de entrada array(n x p)
    #y: array de rótulos (classificação), variável de saída (regressão) array (n x 1)
    #tau: termo de regularização do problema primal do LSSVM (escalar)
    #kernel: string indicando o kernel utilizado ("linear", "polinomial", "gaussiano")
    
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
    
    #--------------------------------------------------------------------------
    #Construção do sistema linear com matriz dos coeficientes
    #simétrica, definda positiva: Ax = B
    #--------------------------------------------------------------------------
    #Construção da matriz A
    H = Omega + (1/tau) * np.identity(n_samples)
    s = np.dot(y, np.linalg.inv(H).dot(y))
    zero_linha = np.zeros((1, n_samples))
    zero_coluna = np.zeros((n_samples, 1))
    A = np.block([[s, zero_linha], [zero_coluna, H]])
    
    #Construção do vetor B
    d1 = 0
    d2 = np.expand_dims(np.ones(n_samples), axis = 1)
    b1 = np.expand_dims(np.array(np.dot(y, np.linalg.inv(H).dot(d2))), axis = 0)
    B = np.concatenate((b1, d2), axis = 0)
    B = np.squeeze(B)
    
    #Aplicação de um método iterativo para a solução do sistema Ax = B
    guess_inicial = np.zeros(n_samples + 1)
    solution = CG_conditioned(A, guess_inicial, B)
    
    #Obtenção do b e dos multiplicadores de Lagrange
    b = solution[0]
    alphas = solution[1:] - np.linalg.inv(H).dot(y) * b
    
    resultado = {'b': b,
                 "mult_lagrange": alphas,
                 "kernel": K}
    
    return resultado


def fit_regre(X, y, tau, kernel):
    #Inputs
    #X: array das variáveis de entrada array(n x p)
    #y: array de rótulos (classificação), variável de saída (regressão) array (n x 1)
    #tau: termo de regularização do problema primal do LSSVM (escalar)
    #kernel: string indicando o kernel utilizado ("linear", "polinomial", "gaussiano")
    
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
    Omega = K
    
    #--------------------------------------------------------------------------
    #Construção do sistema linear com matriz dos coeficientes
    #simétrica, definda positiva: Ax = B
    #--------------------------------------------------------------------------
    #Construção da matriz A
    H = Omega + (1/tau) * np.identity(n_samples)
    um_coluna = np.ones((n_samples))
    s = np.dot(um_coluna, np.linalg.inv(H).dot(um_coluna))
    zero_linha = np.zeros((1, n_samples))
    zero_coluna = np.zeros((n_samples, 1))
    A = np.block([[s, zero_linha], [zero_coluna, H]])
    
    #Construção do vetor B
    d1 = 0
    d2 = np.expand_dims(y, axis = 1)
    b1 = np.expand_dims(np.array(np.dot(um_coluna, np.linalg.inv(H).dot(y))), axis = 0)
    B = np.concatenate((np.expand_dims(b1, axis = 1), d2), axis = 0)
    B = np.squeeze(B)
    
    #Aplicação de um método iterativo para a solução do sistema Ax = B
    x_inicial = np.zeros(n_samples + 1)
    solution = CG_conditioned(A, x_inicial ,B)
    
    #Obtenção do b e dos multiplicadores de Lagrange
    b = solution[0]
    alphas = solution[1:] - np.linalg.inv(H).dot(um_coluna) * b
    
    resultado = {'b': b,
                 "mult_lagrange": alphas,
                 "kernel": K}
    
    return resultado


    
#------------------------------------------------------------------------------
#Implementação do método predict() para problemas de classificação e regressão
#utilizando o CG de Hestenes-Stiefel
#fontes: 
#    -SUYKENS, Johan AK; VANDEWALLE, Joos. Least squares support vector machine classifiers.
#     Neural processing letters, v. 9, p. 293-300, 1999.
#
#    -GOLUB, Gene H.; VAN LOAN, Charles F. Matrix computations. JHU press, 2013.   
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

def predict_regre(alphas, b, kernel, X_treino, X_teste):
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
        estimado[i] = np.sum(np.multiply(alphas, K[i])) + b
    
    return estimado


#------------------------------------------------------------------------------
#Realizando um pequeno teste
#Dataset sintético classificação
#------------------------------------------------------------------------------
style.use("fivethirtyeight")
 
X, y = make_blobs(n_samples = 300, centers = 2, 
               cluster_std = 1, n_features = 2)
 


plt.scatter(X[:, 0], X[:, 1], s = 40, color = 'g')
plt.xlabel("X")
plt.ylabel("Y")
X[1]
plt.show()
plt.clf()

for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1

#Divisão treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3,
                                                        random_state= 49)

resultado = fit_LSSVM(X_treino, y_treino, 0.5, "gaussiano")
alphas = resultado['mult_lagrange']
b = resultado['b']
lssvm_hat = predict_class(alphas, b, "gaussiano", X_treino, y_treino, X_teste)
accuracy_score(y_teste, lssvm_hat)

#------------------------------------------------------------------------------
#Realizando um pequeno teste
#Dataset sintético regressão
#------------------------------------------------------------------------------
X, y = make_regression(n_samples=300, n_features=2, noise=1, random_state=42)

#Divisão treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3,
                                                        random_state=42)


resultado = fit_regre(X_treino, y_treino, 0.5, "gaussiano")
alphas = resultado['mult_lagrange']
b = resultado['b']
lssvm_hat = predict_regre(alphas, b, "gaussiano", X_treino, X_teste)
r2_score(y_teste, lssvm_hat)
np.sqrt(mean_squared_error(y_teste, lssvm_hat))
