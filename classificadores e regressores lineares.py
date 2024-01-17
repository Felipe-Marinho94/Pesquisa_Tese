"""
Implementaçção de alguns métodos de alguns classificadores lineares
Autor:Felipe Pinto Marinho
Data:11/01/2024
"""

#Importando alguns pacotes relevantes
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random as rd
from sklearn import datasets
from sklearn.metrics import accuracy_score
import cvxpy as cp

#Função sinal
def sinal(x):
    
    #Verificação do sinal
    if x > 0:
        return 1
    else:
        return -1


#Perceptron Simples (Forma Primal)
def perceptron(X, Y, eta, K):
    '''
    Dado um conjunto de treinamento linearmente separável se extrai
    X: Matriz de atributos de dimensão (N x p)
    Y: Vetor de rótulos de comprimento (N)
    eta: Taxa de aprendizado
    K: Número Máximo de iterações
    '''
    #Inicialização
    peso_inicial = np.zeros(X.shape[1])
    bias_inicial = 0
    w = peso_inicial
    b = bias_inicial
    
    #Laço externo de iteração
    for k in range(K+1):
        
        #Laço interno de atualização
        for i in range(X.shape[0]):
            
            #Verificação de erros de classificação
            if Y[i]*(np.dot(w, X[i]) + b) <= 0:
                
                #Atualização
                w =+ eta*Y[i]*X[i]
                b =+ eta*Y[i]*X[i]
    
    resultados = {'Peso ótimo': w,
                  'Bias ótimo': b}

    return resultados


'''
Implementação do SVM: caso linear
Considerando um Dataset linearmente separável
'''
#Dataset sintético
X, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, center_box=(0, 10))
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'g^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.show()

for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1
    else:
        y[i] = 1

#solver para o problema baseado no cvxpy
w = cp.Variable(2)
b = cp.Variable(1)
prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(w, np.eye(2))), 
                  [np.multiply(y, X @ w + np.ones([100])*b) >= np.ones([100])])
    
prob.solve()


print("\nO valor ótimo é", prob.value)
print("Uma soluçãové")
print([w.value, b.value])
print("Os multiplicadores de Lagrange para a restrições são")
print(prob.constraints[0].dual_value)

#Função de predict
def predict(X, w, b):
    estimada = []
    for i in range(X.shape[0]):
        estimada.append(sinal(np.dot(X[i, ], w) + b))
    return estimada

#Avaliando o ajuste do SVM
y_hat = predict(X, w.value, b.value)
accuracy_score(y, y_hat)
