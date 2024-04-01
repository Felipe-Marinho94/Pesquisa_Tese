"""
Implementação do algoritmo Alternating Directions Multipliers Method
Data:01/04/2024
"""

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log

#------------------------------------------------------------------------------
#Operador de seft threshold
#------------------------------------------------------------------------------
def Sthresh(x, gamma):
    return np.sign(x)*np.maximum(0, np.absolute(x)-gamma/2.0)


#------------------------------------------------------------------------------
#Função para algoritmo ADMM
#------------------------------------------------------------------------------
def ADMM(A, y):

    m, n = A.shape
    w, v = np.linalg.eig(A.T.dot(A))
    MAX_ITER = 10000

    #Inicialização
    xhat = np.zeros([n, 1])
    zhat = np.zeros([n, 1])
    u = np.zeros([n, 1])

    #Calcula os coeficientes de regressão
    l = sqrt(2*log(n, 10))
    rho = 1/(np.amax(np.absolute(w)))

    #Pré-calcula algumas multiplicações salvas
    AtA = A.T.dot(A)
    Aty = A.T.dot(y)
    Q = AtA + rho*np.identity(n)
    Q = np.linalg.inv(Q)

    i = 0

    while(i < MAX_ITER):

        #Passo de minimização utilizando OLS
        xhat = Q.dot(Aty + rho*(zhat - u))

        #Minimização de z via soft-thresholding
        zhat = Sthresh(xhat + u, l/rho)

        #Atualização dos multiplicadores
        u = u + xhat - zhat

        i = i+1
    return zhat, rho, l


#------------------------------------------------------------------------------
#Realização de alguns testes
#------------------------------------------------------------------------------
A = np.random.randn(50, 200)

num_non_zeros = 10
positions = np.random.randint(0, 200, num_non_zeros)
amplitudes = 100*np.random.randn(num_non_zeros, 1)
x = np.zeros((200, 1))
x[positions] = amplitudes

y = A.dot(x) + np.random.randn(50, 1)

xhat, rho, l = ADMM(A, y)

plt.plot(x, label='Original')
plt.plot(xhat, label = 'Estimativa')

plt.legend(loc = 'upper right')

plt.show()