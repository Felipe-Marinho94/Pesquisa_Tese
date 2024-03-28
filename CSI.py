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
def CSI(X, Y, kernel, rank, kappa = 0.9, delta = 40, tol = 1e-5, centering = True):
    
    #--------------------------------------------------------------------------
    #INPUTS
    #--------------------------------------------------------------------------
    #X: Matriz de dados (array de dim n x p)
    #Y: Matriz de saída (array de dim n x d)
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
    d = Y.shape[1]
    
    if n != Y.shape[0]:
        raise Exception("Dimensões da matriz de entrada e saída não concordam!")
    
    #Inicializações
    m = rank
    