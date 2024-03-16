"""
Implementação dos métodos fit() e predict() para o algoritmo
LSSVM utilzando método do Gradiente Conjugado de Hestens-Stiefel
Data:13/04/2024
"""

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from numpy import linalg
from sklearn.datasets import make_blobs
from matplotlib import style
from Gradiente conjugado import CG

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
#Implementação do método fit() utilizando o CG de Hestenes-Stiefel
#fontes: 
#    -SUYKENS, Johan AK; VANDEWALLE, Joos. Least squares support vector machine classifiers.
#     Neural processing letters, v. 9, p. 293-300, 1999.
#
#    -GOLUB, Gene H.; VAN LOAN, Charles F. Matrix computations. JHU press, 2013.   
#------------------------------------------------------------------------------
def fit(X, y, tau, kernel):
    #Inputs
    #X: array das variáveis de entrada array(n x p)
    #y: array de rótulos (classificação), variável de saída (regressão) array (n x 1)
    #tau: termo de regularização do problema primal do LSSVM (escalar)
    #kernel: string indicando o kernel utilizado ("linear", "polinomial", "gaussiano")
    
    n_samples, n_feautres = X.shape
    
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
    d2 = np.expand_dims(np.ones(100), axis = 1)
    b1 = np.expand_dims(np.array(np.dot(y, np.linalg.inv(H).dot(y))), axis = 0)
    B = np.concatenate((np.expand_dims(b1, axis = 1), d2), axis = 0)
    B = np.squeeze(B)
    
    #Aplicação de um método iterativo para a solução do sistema Ax = B
    solution = CG(A, B, 0.1)
    
    #Obtenção do b e dos multiplicadores de Lagrange
    b = solution[0]
    alphas = solution[1:] - np.linalg.inv(H).dot(y) * b
    
    resultado = {'b': b,
                 "mult_lagrange": alphas,
                 "kernel": K}
    
    return resultado
    

def predict_class(alphas, b, K, X):
    estimado = []
    for i in range(X.shape[0]):
        
    
    return np.sign


#------------------------------------------------------------------------------
#Realizando um pequeno teste
#Dataset sintético
#------------------------------------------------------------------------------
style.use("fivethirtyeight")
 
X, y = make_blobs(n_samples = 100, centers = 3, 
               cluster_std = 1, n_features = 2)
 
plt.scatter(X[:, 0], X[:, 1], s = 40, color = 'g')
plt.xlabel("X")
plt.ylabel("Y")
 
plt.show()
plt.clf()

resultado = fit(X, y, 0.5, "gaussiano")
H = O + (1/100) * np.identity(100)

s = np.array(np.dot(y, np.linalg.inv(H).dot(y)))
s = np.expand_dims(s, axis = 0)
zero_linha = np.zeros((1, 100))
zero_coluna = np.zeros((100, 1))
A = np.block([[s, zero_linha], [zero_coluna, H]])
d2 = np.expand_dims(np.ones(100), axis = 1)
b1 = np.expand_dims(np.array(np.dot(y, np.linalg.inv(H).dot(y))), axis = 0)
B = np.concatenate((np.expand_dims(b1, axis = 1), d2), axis = 0)
B = np.squeeze(B)
solution = CG(A, B, 0.1)
b = solution[0]
alphas = solution[1:] - np.linalg.inv(H).dot(y) * b
