"""
Implementação de nova proposta de poda iterativa para
Esparsificação do conjunto de vetores suportes do modelo LSSVM
Data:29/07/2024
"""

#----------------------------------------------------------------
#Carregando alguns pacotes
#----------------------------------------------------------------
import numpy as np
from numpy import linalg
import math as mt
import matplotlib.pyplot as plt
import seaborn as sns
from kernel import linear_kernel, polinomial_kernel, gaussiano_kernel
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#----------------------------------------------------------------
#Função para o cálculo da matriz A e vetor b para a proposta
#Reduced Set Fixed Sized Levenberg-Marquardt LSSVM (RFSLM-LSSVM)
#----------------------------------------------------------------
def Construct_A_b(X, y, kernel, tau):
    """
    Interface do método
    Ação: Este método ecapsular o cálculo da matriz A e vetor b para
    a proposta RFSLM-LSSVM.

    INPUT:
    X: Matriz de features (array N x p);
    y: Vetor de target (array N x 1);
    kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
    #tau: Parâmetro de regularização do problema primal do LSSVM.

    OUTPUT:
    dois array's representando a matriz A e b, respectivamente.
    """

    #Construção da matriz A e vetor b e inicialização aleatória
    #do vetor de multiplicadores de Lagrange
    
    n_samples, n_features = X.shape
    
    #Matriz de Gram
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            
            #Kernel trick
            if kernel == "linear":
                K[i, j] = linear_kernel(X[i], X[j])
            
            if kernel == "gaussiano":
                K[i, j] = gaussiano_kernel(X[i], X[j])
            
            if kernel == "polinomial":
                K[i, j] = polinomial_kernel(X[i], X[j])
    
    #Construção da Matriz Omega
    Omega = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            Omega[i, j] = y[i] * y[j] * K[i, j]
    
    #Construção da Matriz A
    H = Omega + (1/tau) * np.identity(n_samples)
    A = np.block([[np.array([0]), np.expand_dims(y, axis = 1).T],
                   [np.expand_dims(y, axis = 1), H]])

    #Construção do Vetor B
    B = np.concatenate((np.expand_dims(np.zeros([1]), axis=1),
                         np.expand_dims(np.ones(n_samples), axis = 1)), axis=0)
    
    #Resultados
    resultados = {'A': A,
                  "B": B}
    
    return(resultados)

 
#----------------------------------------------------------------
#Implementação da Proposta 
#Reduced Set Fixed Sized Levenberg-Marquardt LSSVM (RFSLM-LSSVM)
#Source:https://www.researchgate.net/publication/377073903_New_Iterative_Pruning_Methods_for_Least_Squares_Support_Vectors_Machines
#----------------------------------------------------------------
#Implementação do Método Fit() para a proposta FSLM-LSSVM

def fit_RFSLM_LSSVM(X, y, Percent, mu, Red, N, kernel, tau, phi, epsilon):
    """
    Interface do método
    Ação: Este método visa realizar a poda iterativa dos vetores
    de suporte estimados no LSSVM, trabalhando com a matriz dos
    coeficientes reduzida em cada iteração.

    INPUT:
    X: Matriz de features (array N x p);
    y: Vetor de target (array N x 1);
    mu: Taxa de aprendizado (escalar);
    Percent: Porcentagem de redução por iteração de corte (escalar);
    Red: Percentual de remoção (escalar);
    N: Número máximo de iterações (escalar);
    kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
    #tau: iterações de corte (escalar).
    #epsilon: Tolerância (Critério de parada)
    #phi: Regularização do LSSVM

    OUTPUT:
    vetor esparso de multiplicadores de Lagrange estimados. 
    """

    #Gerando os indices para os vetores de suporte
    indices = np.arange(0, X.shape[0])

    #Convertendo array de entrada em Dataframe
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    #Inicialização aleatória do conjunto com vetores de suporte
    X_rest, X_vs, y_rest, y_vs, indices_rest, indices_vs = train_test_split(X, y, indices, test_size= 1-Red)
    X_vs = np.array(X_vs)
    y_vs = np.squeeze(np.array(y_vs))

    #Inicialização aleatória do vetor solução do método de Levenberg-Marquardt
    z_inicial = np.random.normal(loc=0, scale=1, size=(X_vs.shape[0] + 1, 1))
    z = z_inicial

    #Listas para armazenamento
    erros = []
    mult_lagrange = []
    idx_suporte = [range(0, X.shape[0])]

    #Loop de iteração
    for k in range(1, N+1):
        
        #Cálculo da matriz A e vetor b para o conjunto de vetores suporte
        A = Construct_A_b(X_vs, y_vs, kernel, phi)['A']
        B = Construct_A_b(X_vs, y_vs, kernel, phi)['B']

        #Cálculo do erro para a k-ésima iteração
        erro = B - np.matmul(A, z)

        #Atualização
        z_anterior = z
        z = z + np.matmul(linalg.inv(np.matmul(A.T, A) + mu * np.diag(np.matmul(A.T, A))), np.matmul(A.T, erro))

        #Condição para manter a atualização
        erro_novo = B - np.matmul(A, z)
        
        if np.mean(erro_novo**2) < np.mean(erro**2):
            z = z
        else:
            mu = mu/10
            z = z_anterior
        
        if k in np.arange(tau, N, tau, dtype='int'):

            #Número de vetores suporte removidos dados pela porcentagem de redução por iteração de corte
            numero_vetores_suporte_removidos = int(Percent * X_vs.shape[0])

            #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
            idx_remover = np.argsort(np.abs(np.squeeze(z[:len(z)-1])))[:numero_vetores_suporte_removidos]

            #Remoção dos vetores suporte de X_vs, multiplicadores de z e índices de vetores de suporte
            vetores_suporte_removidos = X_vs[idx_remover]
            X_vs = np.delete(X_vs, idx_remover, axis = 0)
            y_vs = np.delete(y_vs, idx_remover)
            z = np.delete(z, idx_remover, axis = 0)

            #Remoção dos indices relacionados aos vetores de suporte
            indices_removidos = indices_vs[idx_remover]
            indices_vs = np.delete(indices_vs, idx_remover)

            #Seleção aleatória de Percent dados para serem
            #adicionados ao conjunto de vetores de suporte
            X_add, X_permanece, y_add, y_permanece, indices_add, indices_permanece = train_test_split(X_rest, y_rest, indices_rest, train_size=numero_vetores_suporte_removidos)
            X_vs = np.append(X_vs, X_add, axis = 0)
            y_vs = np.append(y_vs, y_add)
            indices_vs = np.append(indices_vs, indices_add)

            #Atualização do conjunto de vetores que não são de suporte
            #Remoção
            X_rest = X_rest.loc[~X_rest.index.isin(np.squeeze(indices_add))]
            X_rest.shape
            indices_rest = np.setdiff1d(indices_rest, indices_add)
            len(indices_rest)

            #Adição
            X_rest = pd.concat((X_rest, pd.DataFrame(vetores_suporte_removidos,
                                                      index = indices_removidos)), axis = 0)
            X_rest.shape
            indices_rest = np.append(indices_rest, indices_removidos)
            len(indices_rest)

            #Inicialização aleatória do vetor z_t P%
            z_percent = np.random.normal(loc=0, scale=1, size=(numero_vetores_suporte_removidos, 1))

            #Atualização dos multiplicadores de Lagrange
            z = np.append(z, z_percent, axis=0)
            
        
        #Armazenando o erro e os indices de vetores de suporte
        erros.append(np.mean(erro**2))
        idx_suporte.append(indices_vs)

        #Critério de parada
        if np.abs(np.mean(erro_novo**2)) < epsilon:
            break


    mult_lagrange = np.zeros(X.shape[0])
    mult_lagrange[indices_vs] = np.squeeze(z)[1:len(z)]
    
    #Resultados
    resultados = {"mult_lagrange": mult_lagrange,
                  "b": np.squeeze(z)[0],
                  "Erros": erros,
                  "Indices_multiplicadores": idx_suporte}
    
    return(resultados)





#Realização de alguns testes
X = np.random.normal(0, 1 , size=(1000, 5))
y = np.random.normal(0, 1, 1000)
resultados = fit_RFSLM_LSSVM(X, y, 0.1, 0.01, 0.3, 100, 'gaussiano', 5, 2, 0.001)
resultados['Erros']
sns.lineplot(resultados, x=range(0,len(resultados['Erros'])), y = resultados['Erros'])
plt.xlabel('Iteração')
plt.ylabel('Erro Médio Quadrático (MSE)')
plt.show()

sns.set_theme(style="white")
len(resultados['Indices_multiplicadores'])
index = resultados['Indices_multiplicadores'][0]
index.shape
resultados['Indices_multiplicadores']
fig = plt.subplot(2, 2, 1)
fig.scatter(x=X[:,1], y=X[:,2])
fig.plot(X[index,1], X[index,2], "or")

index = resultados['Indices_multiplicadores'][5]
index.shape
fig1 = plt.subplot(2, 2, 2)
fig1.scatter(x=X[:,1], y=X[:,2])
fig1.plot(X[index,1], X[index,2], "or")

index = -resultados['Indices_multiplicadores'][10]
index.shape
fig2 = plt.subplot(2, 2, 3)
fig2.scatter(x=X[:,1], y=X[:,2])
fig2.plot(X[index,1], X[index,2], "or")

index = -resultados['Indices_multiplicadores'][16]
index.shape
fig3 = plt.subplot(2, 2, 4)
fig3.scatter(x=X[:,1], y=X[:,2])
fig3.plot(X[index,1], X[index,2], "or")

plt.show()
np.array([1, 2, 3])



Red = 0.5
 #Percentual de treino que representa os vetores de suporte
Fed = 1 - Red

#Gerando os indices para os vetores de suporte
indices = np.arange(0, X.shape[0])

#Inicialização aleatória do conjunto com vetores de suporte
X_rest, X_vs, y_rest, y_vs, indices_rest, indices_vs = train_test_split(X, y, indices, test_size= Fed)
X_vs = np.array(X_vs)
y_vs = np.squeeze(np.array(y_vs))

#Inicialização aleatória do vetor solução do método de Levenberg-Marquardt
z_inicial = np.random.normal(loc=0, scale=1, size=(X_vs.shape[0]+1, 1))
z = z_inicial
z

#Listas para armazenamento
erros = []
mult_lagrange = []
Percent = 0.5
mu=0.5

#Loop de iteração
for k in range(1, 20+1):
        
        #Cálculo da matriz A e vetor b para o conjunto de vetores suporte
        A = Construct_A_b(X_vs, y_vs, 'gaussiano', 2)['A']
        B = Construct_A_b(X_vs, y_vs, 'gaussiano', 2)['B']

        #Cálculo do erro para a k-ésima iteração
        erro = B - np.matmul(A, z)

        #Atualização
        z_anterior = z
        z = z + np.matmul(linalg.inv(np.matmul(A.T, A) + 0.1 * np.diag(np.matmul(A.T, A))), np.matmul(A.T, erro))

        #Condição para manter a atualização
        erro_novo = B - np.matmul(A, z)
        if np.mean(erro_novo**2) < np.mean(erro**2):
            z = z
        else:
            mu = mu/10
            z = z_anterior
    
        if k in np.arange(5, 20, 5, dtype='int'):

            #Número de vetores suporte removidos dados pela porcentagem de redução por iteração de corte
            numero_vetores_suporte_removidos = int(Percent * X_vs.shape[0])

            #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
            idx_remover = np.argsort(np.abs(np.squeeze(z[:len(z)-1])))[:numero_vetores_suporte_removidos]
            print(idx_remover)
            len(idx_remover)
            print(X_vs.shape)
            
            #Remoção dos vetores suporte de X_vs, multiplicadores de z e índices de vetores de suporte
            vetores_suporte_removidos = X_vs[idx_remover]
            vetores_suporte_removidos.shape[0]
            X_vs = np.delete(X_vs, idx_remover, axis = 0)
            y_vs = np.delete(y_vs, idx_remover)
            z = np.delete(z, idx_remover, axis = 0)
            print((X_vs.shape[0], len(y_vs), z.shape[0]))
            
            #Remoção dos indices relacionados aos vetores de suporte
            indices_removidos = indices_vs[idx_remover]
            indices_vs = np.delete(indices_vs, idx_remover)
            len(indices_vs)
            print('comprimento de indices removidos', len(indices_removidos))

            #Seleção aleatória de Percent dados para serem
            #adicionados ao conjunto de vetores de suporte
            X_add, X_permanece, y_add, y_permanece, indices_add, indices_permanece = train_test_split(X_rest, y_rest, indices_rest, train_size=Percent)
            X_vs = np.append(X_vs, X_add, axis = 0)
            y_vs = np.append(y_vs, y_add)
            print((X_vs.shape[0], len(y_vs)))
            indices_vs = np.append(indices_vs, indices_add)
            len(indices_vs)
            print('comprimento de indices adicionados', len(indices_add))

            #Atualização do conjunto de vetores que não são de suporte
            #Remoção
            X_rest = X_rest.loc[~X_rest.index.isin(np.squeeze(indices_add))]
            X_rest.shape
            indices_rest = np.setdiff1d(indices_rest, indices_add)
            len(indices_rest)

            #Adição
            X_rest = pd.concat((X_rest, pd.DataFrame(vetores_suporte_removidos,
                                                      index = indices_removidos)), axis = 0)
            X_rest.shape
            indices_rest = np.append(indices_rest, indices_removidos)
            len(indices_rest)

            #Inicialização aleatória do vetor z_t P%
            z_percent = np.random.normal(loc=0, scale=1, size=(numero_vetores_suporte_removidos, 1))

            #Atualização dos multiplicadores de Lagrange
            z = np.append(z, z_percent, axis=0)
            z.shape
            


len(np.setdiff1d(indices_rest, indices_add))
#Número de vetores suporte removidos dados pela porcentagem de redução por iteração de corte
numero_vetores_suporte_removidos = int(Percent * X_vs.shape[0])
numero_vetores_suporte_removidos

#Ordenar os menores valores absolutos dos multiplicadores de Lagrange
idx_remover = np.argsort(np.abs(np.squeeze(z)))[:(numero_vetores_suporte_removidos)]
idx_remover
len(idx_remover)

#Remoção dos vetores suporte de X_vs, multiplicadores de z e índices de vetores de suporte
X_vs.shape
vetores_suporte_removidos = X_vs[idx_remover]
vetores_suporte_removidos.shape[0]
X_vs = np.delete(X_vs, idx_remover, axis = 0)
X_vs.shape[0]
z = np.delete(z, idx_remover, axis = 0)
z.shape[0]
indices_removidos = indices_vs[idx_remover]
indices_removidos
indices_vs = np.delete(indices_vs, idx_remover)
len(indices_vs)

#Seleção aleatória de Percent dados para serem
#adicionados ao conjunto de vetores de suporte
X_rest.shape
X_add, X_permanece, y_add, y_permanece, indices_add, indices_permanece = train_test_split(X_rest, y_rest, indices_rest, train_size=Percent)
X_vs = np.append(X_vs, X_add, axis = 0)
X_vs.shape
indices_vs = np.append(indices_vs, indices_add)
len(indices_vs)

#Atualização do conjunto de vetores que não são de suporte
#Remoção
X_rest = X_rest.loc[~X_rest.index.isin(np.squeeze(indices_add))]
X_rest.shape

#Adição
pd.DataFrame(vetores_suporte_removidos)
X_rest = pd.concat((X_rest, pd.DataFrame(vetores_suporte_removidos, index = indices_removidos)), axis = 0)
X_rest.shape
for i in X_rest.index:
    print(i)

len(np.unique(X_rest.index))
#Inicialização aleatória do vetor z_t P%
z_percent = np.random.normal(loc=0, scale=1, size=(numero_vetores_suporte_removidos))

#Atualização dos multiplicadores de Lagrange
z = np.append(z, z_percent)
z.shape
len(indices_vs)


np.squeeze(z)