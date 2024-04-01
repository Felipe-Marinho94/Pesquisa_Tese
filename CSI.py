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
    P = np.r_[0, 1:n, n] #Pivots
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
                jast += (kadv - 1)
                
                
                #Permutação dos indices
                P[[kadv, jast]] = P[[jast, kadv]] 
                Dadv[[kadv, jast]] = Dadv[[jast, kadv]]
                D[[kadv, jast]] = D[[jast, kadv]]
                A1[[kadv, jast]] = A1[[jast, kadv]]
                G[[kadv, jast], 0:(kadv-1)] = G[[jast, kadv], 0:(kadv-1)]
                Q[[kadv, jast], 0:(kadv-1)] = Q[[jast, kadv], 0:(kadv-1)]
                
                #Calculando a nova coluna de Cholesky
                G[kadv,kadv] = Dadv[kadv]
                G[kadv,kadv] = np.sqrt(G[kadv,kadv])
                newKcol = K[P[kadv + 1: n], P[kadv]]
                G[(kadv+1):n, kadv] = 1/(G[kadv, kadv] * (newKcol - G[(kadv + 1): n, 1: (kadv - 1) ] * 
                                                          G[kadv, 1: (kadv - 1)].T))
                
                #Atualização da diagonal
                Dadv[(kadv+1):n] -= G[(kadv+1):n, kadv]**2
                Dadv[kadv] = 0
                
                #Realizando as decomposições QR
                if centering:
                    Gcol = G[:, kadv] - np.apply_along_axis(np.mean, 1, G)
                else:
                    Gcol = G[:, kadv]
                
                R[1:(kadv-1), kadv] = Q[:, 1:(kadv-1)].T * Gcol
                Q[:, kadv] = Gcol - Q[:, 1:(kadv-1)] * R[1:(kadv-1), kadv]
                R[kadv, kadv] = np.norm(Q[:, kadv])
                
                #Atualização das quantidades em cachê
                if centering:
                    GTG[1:kadv, kadv] = G[:, kadv].T * G[:, kadv]
                else:
                    GTG[1:kadv, kadv] = R[1:kadv, 1:kadv].T * R[1:kadv, kadv]
                
                GTG[kadv, 1:kadv] = GTG[1:kadv, kadv]
                QTy[kadv, :] = Q[:, kadv].T * y[P, :]
                QTyyTQ[kadv, 1:kadv] = QTy[kadv, :] * QTy[1:kadv, :].T
                QTyyTQ[1:kadv, kadv] = QTyyTQ[kadv, 1:kadv].T
                
                #Atualizando os custos
                A1[kadv:n] = A1[kadv:n] + GTG[kadv,kadv] * ( G[kadv:n,kadv]**2 )
                A1[kadv:n] = A1[kadv:n] + 2 * G[kadv:n,kadv] * ( G[kadv:n,1:kadv-1] * GTG[1:kadv-1,kadv] )


    #Calculando os custos restantes
    A2 = np.sum((G[:, 1:kadv] * (R[1:kadv, 1:kadv].T * QTy[1:kadv, :]))**2, axis=1)
    A3 = np.sum((G[:, 1:kadv] * R[1:kadv, 1:kadv].T)**2, axis = 1)
    
    #Laço principal
    while k < m:
        k += 1
        