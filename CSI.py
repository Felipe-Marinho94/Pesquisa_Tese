"""
Implementação da decomposição de Cholesky Side Information (CSI)
Data:28/03/2024
"""

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sns

#------------------------------------------------------------------------------
#Construção do método CSI
#------------------------------------------------------------------------------
def CSI(X, y, K, rank, kappa = 0.9, delta = 40, tol = 1e-5, centering = True):
    
    #--------------------------------------------------------------------------
    #INPUTS
    #--------------------------------------------------------------------------
    #X: Matriz de dados (array de dim n x p)
    #y: Matriz de saída (array de dim n x d)
    #K: Matriz de Gram (array de dim n x n)
    #rank: posto maximal
    ### kappa : controla o trade-off entre aproximação de K e predição de Y (sugerido: .99)
    ## centering : 1 se normaliza, 0 caso contrário (sugerido: 1)
    ## delta : n° de colunas de Cholesky para realizar em avanço (sugerido: 40)
    ## tol : ganho mínimo por iteração (sugerido: 1e-4)
    
    ## OUTPUT
    ## G : Decomposição de Cholesky -> K(P,P) é aproximada por G*G'
    ## P : Matriz de Permutação
    ## Q,R : Decomposição QR de G (ou normalização(G) se centering = True)
    ## error1 : tr(K-G*G')/tr(K) a cada passo da decomposição
    ## error2 : ||y-Q*Q'*y||.F^2 / ||y||.F^2 a cada passo da decomposição
    ## predicted.gain : ganho predito antes da adição de colunas
    ## true.gain : ganho atual após a adição de cada coluna
    
    n = X.shape[0]
    d = y.shape[1]
    
    if n != y.shape[0]:
        raise Exception("Dimensões da matriz de entrada e saída não concordam!")
    
    #Inicializações
    m = rank
    
    #garantindo que m seja menor do que n
    m = min(m, n-2)
    G = np.zeros((n, min(m + delta, n)))
    diagK = np.diag(K)
    P = range(n) #Pivots
    Q = np.zeros((n, min(m + delta, n))) #matriz Q na decomposição QR
    R = np.zeros((min(m + delta, n), min(m + delta, n))) #matriz R na decomposição QR
    traceK = sum(diagK)
    lamb = (1 - kappa)/traceK
    
    if centering:
        y = y - np.apply_along_axis(np.mean, 1, y)
    
    sumy2 = sum(y**2)
    mu = kappa/sumy2
    error1 = traceK
    error2 = sumy2
    truegain = np.zeros((min(m + delta, n)))
    predictedgain = truegain
    
    k = 0  #indice atual da decomposição de Cholesky
    kadv = 0 #indice atual dos passos look aheads
    
    Dadv = diagK
    D = diagK
    
    #Garantir que delta seja menor do que n-2
    delta = min(delta, n - 2)
    
    #Custo de aproximações em catched
    A1 = np.zeros((n,1))
    A2 = np.zeros((n,1))
    A3 = np.zeros((n,1))
    GTG = np.zeros((m + delta, m + delta))
    QTy = np.zeros((m + delta, d))
    QTyyTQ = np.zeros((m + delta, m + delta))
    
    #Realiza delta decomposições QR e Cholesky
    if delta > 0:
        for i in range(delta):
            kadv += 1
            
            #Seleciona o melhor índice
            diagmax = Dadv[kadv]
            jast = 1
            
            for j in range(n - kadv + 1):
                if Dadv[j + kadv - 1] > diagmax/.99:
                    diagmax = Dadv[j + kadv - 1]
                    jast = j
            
            if diagmax < 1e12:
                kadv = kadv - 1
                break
            else:
                



