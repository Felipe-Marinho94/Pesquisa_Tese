#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_regression
from matplotlib import style
from kernel import predict_class, fit_class, fit_regre, predict_regre
from LSSVM_ADMM import fit_LSSVM_ADMM, predict_class_LSSVM_ADMM
from FSLM_LSSVM import fit_FSLM_LSSVM
from RFSLM_LSSVM import fit_RFSLM_LSSVM
from métricas import metricas
from time import process_time

#------------------------------------------------------------------------------
#Realizando um pequeno teste
#Dataset sintético classificação
#------------------------------------------------------------------------------
style.use("fivethirtyeight")
 
X, y = make_blobs(n_samples = 2000, centers = 2, 
               cluster_std = 2, n_features = 2)
 

plt.scatter(X[:, 0], X[:, 1], s = 40, color = 'g')
plt.xlabel("X")
plt.ylabel("Y")
X[1]
plt.show()
plt.clf()

for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1

y
#Divisão treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3,
                                                        random_state= 49)


#Ajuste de modelo LSSVM com sistema KKT resolvido numericamente via
#gradiente conjugado de Hestenes-Stiefel
tic = process_time()
resultado_LSSVM = fit_class(X_treino, y_treino, 0.5, "gaussiano")
toc = process_time()
time_LSSVM = toc - tic 
time_LSSVM

alphas_LSSVM = resultado_LSSVM['mult_lagrange']
alphas_LSSVM
b_LSSVM = resultado_LSSVM['b']
b_LSSVM
LSSVM_hat = predict_class(alphas_LSSVM, b_LSSVM, "gaussiano", X_treino, y_treino, X_teste)
metricas(y_teste, LSSVM_hat)

#Primeira proposta denominada LSSVM-ADMM
tic = process_time()
resultado_LSSVM_ADMM = fit_LSSVM_ADMM(X_treino, y_treino, 0.5, "gaussiano")
toc = process_time()
time_LSSVM_ADMM = toc - tic
time_LSSVM_ADMM

alphas_LSSVM_ADMM = resultado_LSSVM_ADMM['mult_lagrange']
alphas_LSSVM_ADMM
LSSVM_ADMM_hat = predict_class_LSSVM_ADMM(alphas_LSSVM_ADMM, "gaussiano", X_treino, X_teste)
metricas(y_teste, LSSVM_ADMM_hat)
LSSVM_ADMM_hat

#Proposta FSLM_LSSVM
tic = process_time()
resultado_FSLM_LSSVM = fit_FSLM_LSSVM(X_treino, y_treino, 2, 0.3, 0.5, 50, "gaussiano", 2, 0.001)
toc = process_time()
time_FSLM_LSSVM = toc - tic
time_FSLM_LSSVM

alphas_FSLM_LSSVM = resultado_FSLM_LSSVM['mult_lagrange']
alphas_FSLM_LSSVM
b_FSLM_LSSVM = resultado_FSLM_LSSVM['b']
b_FSLM_LSSVM
FSLM_LSSVM_hat = predict_class(alphas_FSLM_LSSVM, b_FSLM_LSSVM, "gaussiano", X_treino, y_treino, X_teste)
metricas(y_teste, FSLM_LSSVM_hat)

#Proposta RFSLM_LSSVM
tic = process_time()
resultado_RFSLM_LSSVM = fit_RFSLM_LSSVM(X_treino, y_treino, 0.1, 0.01, 0.5, 50, 'gaussiano', 5, 2, 0.001)
toc = process_time()
time_RFSLM_LSSVM = toc - tic
time_RFSLM_LSSVM

alphas_RFSLM_LSSVM = resultado_RFSLM_LSSVM['mult_lagrange']
alphas_RFSLM_LSSVM
b_RFSLM_LSSVM = resultado_RFSLM_LSSVM['b']
b_RFSLM_LSSVM
RFSLM_LSSVM_hat = predict_class(alphas_FSLM_LSSVM, b_FSLM_LSSVM, "gaussiano", X_treino, y_treino, X_teste)
metricas(y_teste, RFSLM_LSSVM_hat)


#------------------------------------------------------------------------------
#Realizando um pequeno teste
#Dataset sintético regressão
#------------------------------------------------------------------------------
X, y = make_regression(n_samples=300, n_features=2, noise=1, random_state=42)

#Divisão treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3,
                                                        random_state=42)


resultado = fit_regre(X_treino, y_treino, 0.5, "gaussiano")
alphas = resultado['mult_lagrange']
b = resultado['b']
lssvm_hat = predict_regre(alphas, b, "gaussiano", X_treino, X_teste)
r2_score(y_teste, lssvm_hat)
np.sqrt(mean_squared_error(y_teste, lssvm_hat))
