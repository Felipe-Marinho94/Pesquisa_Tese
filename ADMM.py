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

    "Function to caluculate min 1/2(y - Ax) + l||x||"
    "via alternating direction methods"
    xhat = np.zeros([n, 1])
    zhat = np.zeros([n, 1])
    u = np.zeros([n, 1])

    "Calculate regression co-efficient and stepsize"
    l = sqrt(2*log(n, 10))
    rho = 1/(np.amax(np.absolute(w)))

    "Pre-compute to save some multiplications"
    AtA = A.T.dot(A)
    Aty = A.T.dot(y)
    Q = AtA + rho*np.identity(n)
    Q = np.linalg.inv(Q)

    i = 0

    while(i < MAX_ITER):

        "x minimisation step via posterier OLS"
        xhat = Q.dot(Aty + rho*(zhat - u))

        "z minimisation via soft-thresholding"
        zhat = Sthresh(xhat + u, l/rho)

        "mulitplier update"
        u = u + xhat - zhat

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