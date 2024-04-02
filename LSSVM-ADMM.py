"""
Implementação das propostas para a tese
Data:02/04/2024
"""

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
import math

#------------------------------------------------------------------------------
#Implementação do método Fit() para a primeira proposta
#baseada na esparsificação do LSSVM utilizando regularização L1 (LASSO)
#no problema primal e resolução do problema de otimização pela aplicação
#do algoritmo Alternating Directions Multipliers Method (ADMM)
#------------------------------------------------------------------------------
def fit_LSSVM_ADMM(X, y):
    #Input
    #X:Matriz de Dados (array n x p)
    #y:vetor de rótulos (classificação) 
    #  ou vetor de respostas numéricas (regressão) (array n x 1)
    
    #Output
    #x_ótimo: vetor solução aproximado do sistema KKT Ax = b (array n+1 x 1)
    #x_ótimo = [alphas b].T
    
    #Construção da matriz dos coeficiente A

