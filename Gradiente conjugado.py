'''
Codigo para a tese - Least Square Support Vector Machine
Data:12/03/2024
'''

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
import math

#------------------------------------------------------------------------------
#Implementacao de metodologia para solucao do LSSVM (fit) utilizando o algoritmo
#do Gradiente Conjugado Hestenes-Stiefel
#Fonte:  SUYKENS, Johan AK; VANDEWALLE, Joos.
# Least squares support vector machine classifiers. Neural processing letters,
# v. 9, p. 293-300, 1999.
#------------------------------------------------------------------------------


def CG(A, b, epsilon):
    #--------------------------------------------------------------------------
    #Método iterativo para a solução de sistemas lineares Ax = B, com A sendo
    #simétrica e definida positiva
    #INPUTS:
    #A:matriz dos coeficientes de ordem m x n (array)
    #B:vetor de termos independentes m x 1 (array)
    #epsilon:tolerancia (escalar)
    #OUTPUTS:
    #x*:vetor solucao aproximada n x 1 (array) 
    #--------------------------------------------------------------------------
    
    #Inicialização
    i = 0
    x = np.zeros(A.shape[1])
    r = b - A.dot(x)
    r_anterior = r
    
    while np.sqrt(np.dot(r, r)) > epsilon:
        i += 1
        if i == 1:
            p = r
        else:
            beta = np.dot(r, r)/np.dot(r_anterior, r_anterior)
            p = r + beta * p
        
        lamb = np.dot(r, r)/np.dot(p, A.dot(p))
        x += lamb * p
        r_anterior = r
        r += -lamb * A.dot(p)
        
    return x


#------------------------------------------------------------------------------
#Implementacao de metodologia para solucao do LSSVM (fit) utilizando o algoritmo
#do Gradiente Conjugado precondicionado usando decomposição de Cholesy incompleta
#Fonte:
#http://www.grimnes.no/algorithms/preconditioned-conjugate-gradients-method-matrices/
#------------------------------------------------------------------------------
MAX_ITERATIONS = 10**4
MAX_ERROR = 10**-3


x = np.array([[2,3,4,5]], dtype=float).T
A = np.array([[3,1,0,0],[1,4,1,3],[0,1,10,0],[0,3,0,3]], dtype=float)
b = np.array([[1,1,1,1]], dtype=float).T

#Decomposição de Cholesky imcompleta
def ichol( A ):
    mat = np.copy( A )
    n = mat.shape[1]
    
    for k in range(n):
        mat[k,k] = math.sqrt( mat[k,k] )
        for i in range(k+1, n):
            if mat[i,k] != 0:
                mat[i,k] = mat[i,k] / mat[k,k]
        for j in range(k+1, n):
            for i in range(j, n):
                if mat[i,j] != 0:
                    mat[i,j] = mat[i,j] - mat[i,k] * mat[j,k]
    for i in range(n):
        for j in range(i+1, n):
            mat[i,j] = 0
    
    return mat


def CG_conditioned( A,x,b ):
    residual = b - A.dot(x)
    preconditioner = np.linalg.inv( ichol(A) )
    
    z = np.dot( preconditioner, residual )
    d = z
    
    error = np.dot(residual.T, residual)
    
    iteration = 0
    while iteration<MAX_ITERATIONS and error>MAX_ERROR**2:
        q        = np.dot( A, d )
        a        = np.dot(residual.T, z)/np.dot( d.T, q )
        
        phi      = np.dot( z.T,  residual )
        old_res  = residual
        
        x        = x + a * d
        residual = residual - a * q
        
        z        = np.dot( preconditioner, residual )
        beta     = np.dot(z.T, (residual-old_res))/phi # Polak-Ribiere
        d        = z + beta * d
        
        error    = residual.T.dot(residual)
        
        iteration += 1


    
    if iteration<MAX_ITERATIONS:
        print('Precisão alcançada. Iterações:', iteration)
    else:
        print('Convergência não alcançada.')
        
    return x

#Testando a função
A = np.array([[3, 2],
              [2, 6]])

B = np.array([2, -8])
CG(A, B, 0.01)
x_inicial = np.zeros(2)
CG_conditioned(A, x_inicial, B)

