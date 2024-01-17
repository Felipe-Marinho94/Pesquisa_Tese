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

'''
Implementação de algumas funções relevantes
utilizadas para os modelos SVM e LSSVM
'''
#Função sinal
def sinal(x):
    
    #Verificação do sinal
    if x > 0:
        return 1
    else:
        return -1


#Função de predict
def predict(X, w, b):
    estimada = []
    for i in range(X.shape[0]):
        estimada.append(sinal(np.dot(X[i, ], w) + b))
    return estimada

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
#solver para o problema baseado no cvxpy
def SVM_linear(X, y):
    
    #Inicialização das variáveis
    w = cp.Variable(2)
    b = cp.Variable(1)
    
    #Definição do problema de otimização convexa
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(w, np.eye(2))), 
                      [np.multiply(y, X @ w + np.ones([100])*b) >= np.ones([100])])
    
    #Resolução
    prob.solve()
    
    #Resultados
    resultados = {'w_opt': w.value,
                  'b_opt': b.value,
                  'valor_opt': prob.value,
                  'lagrange_opt': prob.constraints[0].dual_value}
    
    return resultados

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

resultados_SVM_linear = SVM_linear(X, y)

print("\nO valor ótimo é", resultados_SVM_linear['valor_opt'])
print("Uma solução é")
print([resultados_SVM_linear['w_opt'], resultados_SVM_linear['b_opt']])
print("Os multiplicadores de Lagrange para a restrições são")
print(resultados_SVM_linear['lagrange_opt'])

'''
Avaliação dos modelos implementados utilizando
diversas métricas para classificação
'''
#Avaliando o ajuste do SVM linear no dataset lineramente separável
y_hat = predict(X, resultados_SVM_linear['w_opt'], resultados_SVM_linear['b_opt'])
accuracy_score(y, y_hat)
