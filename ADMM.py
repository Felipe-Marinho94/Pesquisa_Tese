"""
Implementação do algoritmo Alternating Directions Nultipliers Method (ADMM)
Data:28/03/2024
"""

#------------------------------------------------------------------------------
#Carregando alguns pacotes
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log

#------------------------------------------------------------------------------
#Função de soft threshold
#------------------------------------------------------------------------------
def Sthresh(x, gamma):
    return np.sign(x)*np.maximum(0, np.absolute(x)-gamma/2.0)

#------------------------------------------------------------------------------
#Alternating Directions Nultipliers Method (ADMM)
#------------------------------------------------------------------------------

def ADMM(A, y):

    m, n = A.shape
    w, v = np.linalg.eig(A.T.dot(A))
    MAX_ITER = 50000

    "Função para calcular min 1/2(y - Ax) + l||x||"
    "via alternating direction methods"
    xhat = np.zeros([n, 1])
    zhat = np.zeros([n, 1])
    u = np.zeros([n, 1])

    "Calcular os coeficientes e tamanho do passo"
    l = sqrt(2*log(n, 10))
    rho = 1/(np.amax(np.absolute(w)))

    "Pre-ca´cula para salva tais operações"
    AtA = A.T.dot(A)
    Aty = A.T.dot(y)
    Q = AtA + rho*np.identity(n)
    Q = np.linalg.inv(Q)

    i = 0

    while(i < MAX_ITER):

        "x minimização via OLS"
        xhat = Q.dot(Aty + rho*(zhat - u))

        "z minimização via soft-thresholding"
        zhat = Sthresh(xhat + u, l/rho)

        "atualização dos multiplicadores"
        u = u + xhat - zhat

        i = i+1
    return zhat, rho, l


#------------------------------------------------------------------------------
#Fast Alternating Directions Nultipliers Method (Fast ADMM)
#source:https://core.ac.uk/download/pdf/83831191.pdf
#------------------------------------------------------------------------------
def Fast_ADMM(A, y, tau):

    m, n = A.shape
    w, v = np.linalg.eig(A.T.dot(A))
    MAX_ITER = 50000

    "Função para calcular min 1/2(y - Ax) + l||x||"
    "via alternating direction methods"
    #Inicialização
    xhat = np.zeros([n, 1])
    
    z = np.zeros([n, 1])
    zhat = np.zeros([n, 1])

    u = np.zeros([n, 1])
    uhat = np.zeros([n, 1])

    alpha = 1 #termo de momento

    "Calcular os coeficientes e tamanho do passo"
    l = sqrt(2*log(n, 10))
    rho = 1/(np.amax(np.absolute(w)))

    "Pre-cálcula para salva tais operações"
    AtA = A.T.dot(A)
    Aty = A.T.dot(y)
    Q = AtA + rho*np.identity(n)
    Q = np.linalg.inv(Q)

    i = 0

    while(i < MAX_ITER):

        "x minimização via OLS"
        xhat = Q.dot(Aty + rho*(zhat - uhat))

        "z minimização via soft-thresholding"
        z_anterior = z
        z = Sthresh(xhat + uhat, l/rho)

        "atualização dos multiplicadores"
        u_anterior = u
        u = uhat + xhat - z

        "atualização do termo de momento"
        alpha_anterior = alpha
        alpha = (1+sqrt(1+4*(alpha**2)))/2

        "Atualização do vetor z"
        zhat = z + (alpha_anterior -1)*(z-z_anterior)/alpha

        "Atualização dos multiplicadores de Lagrange"
        uhat = u + (alpha_anterior -1)*(u-u_anterior)/alpha

        i = i+1
    return zhat, rho, l

#------------------------------------------------------------------------------
#Fast Alternating Directions Nultipliers Method com reinicialização (Fast ADMM restart)
#source:https://core.ac.uk/download/pdf/83831191.pdf
#------------------------------------------------------------------------------
def Fast_ADMM_restart(A, y, tau, neta):

    m, n = A.shape
    w, v = np.linalg.eig(A.T.dot(A))
    MAX_ITER = 50000

    "Função para calcular min 1/2(y - Ax) + l||x||"
    "via alternating direction methods"
    #Inicialização
    xhat = np.zeros([n, 1])
    
    z = np.zeros([n, 1])
    zhat = np.zeros([n, 1])

    u = np.zeros([n, 1])
    uhat = np.zeros([n, 1])

    alpha = 1 #termo de momento
    c = 0

    "Calcular os coeficientes e tamanho do passo"
    l = sqrt(2*log(n, 10))
    rho = 1/(np.amax(np.absolute(w)))

    "Pre-cálcula para salva tais operações"
    AtA = A.T.dot(A)
    Aty = A.T.dot(y)
    Q = AtA + rho*np.identity(n)
    Q = np.linalg.inv(Q)

    i = 0

    while(i < MAX_ITER):

        "x minimização via OLS"
        xhat = Q.dot(Aty + rho*(zhat - uhat))

        "z minimização via soft-thresholding"
        z_anterior = z
        z = Sthresh(xhat + uhat, l/rho)

        "atualização dos multiplicadores"
        u_anterior = u
        u = uhat + xhat - z

        #Regra para reinicialização
        c_anterior = c
        c = (1/tau)*(np.linalg.norm(u - uhat)**2) + tau*(np.linalg.norm(z - zhat)**2)

        alpha_anterior = alpha
        if c < neta*c_anterior:
            "atualização do termo de momento"
            alpha = (1+sqrt(1+(4*alpha**2)))/2

            "Atualização do vetor z"
            zhat = z + (alpha_anterior -1)*(z-z_anterior)/alpha

            "Atualização dos multiplicadores de Lagrange"
            uhat = u + (alpha_anterior -1)*(u-u_anterior)/alpha

        else:
            alpha = 1
            zhat = z_anterior
            uhat = u_anterior
            c = c/neta
        i = i+1
    return zhat, rho, l

#------------------------------------------------------------------------------
#Realização de testes
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
plt.plot(xhat, label = 'Estimate')

plt.legend(loc = 'upper right')

plt.show()

xhat, rho, l = Fast_ADMM(A, y, 0.5)

plt.plot(x, label='Original')
plt.plot(xhat, label = 'Estimate')

plt.legend(loc = 'upper right')

plt.show()

xhat, rho, l = Fast_ADMM_restart(A, y, 0.5, 0.5)

plt.plot(x, label='Original')
plt.plot(xhat, label = 'Estimate')

plt.legend(loc = 'upper right')

plt.show()