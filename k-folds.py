"""
Implementação da metodologia de grid seach com validação cruzada
para Tunning dos modelos em análise
Data:13/03/2024
"""

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from numpy import linalg
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#------------------------------------------------------------------------------
#Carregando um dataset para testar nossa implementação
#fonte:
#https://medium.com/@avijit.bhattacharjee1996/implementing-k-fold-cross-validation-from-scratch-in-python-ae413b41c80d
#------------------------------------------------------------------------------
X, y = make_classification(n_samples = 50,
                                       n_features = 5,
                                       n_informative = 5,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       weights = [0.51, .49])

#------------------------------------------------------------------------------
#Dividindo o dataset em folds
#------------------------------------------------------------------------------
def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds

#Definindo o número de folds (K)
k = 5

#Tomando índices para os folds
fold_indices = kfold_indices(X, k)
fold_indices

#------------------------------------------------------------------------------
#Realizando o procedimento de validação cruzada
#------------------------------------------------------------------------------
#Inicializando o modelo (e.g., Regressão Logistica)
model = LogisticRegression()

#Inicializando a lista para armazenar as métricas de desempenho
scores = []

#Iterando sobre cada Fold
for train_indices, test_indices in fold_indices:
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    #Treinando o modelo no treino
    model.fit(X_train, y_train)
    
    #Realizando predições no conjunto de teste
    y_pred = model.predict(X_test)
    
    #Calcula a acurácia para este fold
    fold_score = accuracy_score(y_test, y_pred)
    
    #Adiciona o valor da métrica na lista
    scores.append(fold_score)

#Calcula a acurácia média atravé dos folds
mean_accuracy = np.mean(scores)
print("Score para cada fold:", scores)
print("Acurácia:", mean_accuracy)
