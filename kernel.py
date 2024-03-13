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
            
            #Kerneel trick
            if kernel == "linear":
                K[i, j] = linear_kernel(X[i], X[j])
            
            if kernel == "gaussiano":
                K[i, j] = gaussiano_kernel(X[i], X[j])
            
            if kernel == "polinomial":
                K[i, j] = polinomial_kernel(X[i], X[j])
    return K


#Realizando um pequeno teste
#Dataset sintético
style.use("fivethirtyeight")
 
X, y = make_blobs(n_samples = 100, centers = 3, 
               cluster_std = 1, n_features = 2)
 
plt.scatter(X[:, 0], X[:, 1], s = 40, color = 'g')
plt.xlabel("X")
plt.ylabel("Y")
 
plt.show()
plt.clf()

K = fit(X, y, 0.5, "gaussiano")
