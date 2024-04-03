"""
Realizando alguns testes e obtnção de resultados
Data:03//04/2024
"""

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
from Kernel import fit_class, fit_regre, predict_class, predict_regre
from LSSVM_ADMM import fit_LSSVM_ADMM
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_regression
from matplotlib import style
from numpy.random import rand
from Gradiente_conjugado import ichol
import numpy as np

#------------------------------------------------------------------------------
#Realizando um pequeno teste
#Dataset sintético classificação
#------------------------------------------------------------------------------
style.use("fivethirtyeight")
 
X, y = make_blobs(n_samples = 300, centers = 2, 
               cluster_std = 2, n_features = 2)
 

c = np.random.randint(1, 2, size=300)
plt.scatter(X[:, 0], X[:, 1], s = 40, c = c)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
plt.clf()

for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1

#Divisão treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3,
                                                        random_state= 49)

resultado_LSSVM = fit_class(X_treino, y_treino, 0.5, "gaussiano")
resultado_LSSVM_ADMM = fit_LSSVM_ADMM(X_treino, y_treino, "gaussiano", 0.5)
alphas = resultado_LSSVM['mult_lagrange']
alphas_lssvm_admm = resultado_LSSVM_ADMM['mult_lagrange']
b = resultado_LSSVM['b']
lssvm_hat = predict_class(alphas, b, "gaussiano", X_treino, y_treino, X_teste)
lssvm_admm_hat = predict_class(alphas, 0, "gaussiano", X_treino, y_treino, X_teste)

accuracy_score(y_teste, lssvm_hat)
accuracy_score(y_teste, lssvm_admm_hat)
