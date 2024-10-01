"""
Implementação de melhoria na proposta Fixed Size Levenberg-Marquardt Least
Square Support Vector Machine (FSLM-LSSVM)
Data: 30/09/2024
"""

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as ln
from kernel import linear_kernel, polinomial_kernel, gaussiano_kernel
from sklearn.datasets import make_blobs, make_regression
from sklearn.cluster import DBSCAN
from matplotlib import style

#------------------------------------------------------------------------------
#Implementação de algumas funções relevantes
#------------------------------------------------------------------------------
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


#Realizando alguns testes
style.use("fivethirtyeight")
 
X, y = make_blobs(n_samples = 2000, centers = 2, 
               cluster_std = 2.5, n_features = 2)
 

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

Construct_A_b(X, y, 'gaussiano', 0.5)

###############################################################################
def purity_index(X, y):
    '''
    Função para determinar o nível de pureza de uma determinado conjunto com
    base nas alterações de sinal do rótulo de cada amostra
    INPUT:
        X - Array de features (Array de dimensão n x p);
        y - Array de targets (Array de dimensão n x 1);
        index - Conjunto de índices correspondentes a um subconjunto de X.
        
    OUTPUT:
        pureza - Nível de pureza dada pelas trocas de sinal no rótulo de cada
        amostra do subconjunto em análise.
    '''
    #Incialização
    contador = 0
    y_anterior = y[0]
    
    for i in range(len(y)):

        if  y[i] * y_anterior < 0:
            contador += 1
        
        y_anterior = y[i]
    
    return(contador)

#Realizando alguns testes
index = np.random.randint(1, 100, size = (100))
y[index[0]]
purity_index(X, y)

###############################################################################
def cluster_optimum(X, y, eps = 0.5):
    '''
    Método para determinação do cluster com maior nível de impureza, onde o 
    processo de clusterização é baseado no algoritmo DBSCAN.
    INPUT:
        X - Array de features (Array de dimensão n x p);
        y - Array de targets (Array de dimensão n x 1);
        eps - máxima distância entre duas amostras para uma ser considerada 
        vizinhança da outra (float, default = 0.5).
        
    OUTPUT:
        índices do cluster maior impureza.
    '''
    
    #Convertendo dataframe
    X = pd.DataFrame(X)
    y = pd.Series(y, name = "y")
    
    #Clusterização utilizando DBSCAN
    clustering = DBSCAN(eps = eps).fit(X)
    
    #Recuperando os índices
    cluster = pd.Series(clustering.labels_, name = "cluster")
    df = pd.concat([cluster, X, y], axis = 1)
    purity = df.groupby('cluster').apply(purity_index, df.y)
    
    return(df.where(df.cluster == purity.idxmax()).dropna(axis = 0).index)

    
#Realizando alguns testes
clustering = DBSCAN().fit(pd.DataFrame(X))
cluster = pd.Series(clustering.labels_, name = "cluster")    
X_new = pd.concat([cluster, pd.DataFrame(X), pd.Series(y, name = "y")],
                  axis = 1)
X_new.head()
purity = X_new.groupby('cluster', group_keys = True).apply(purity_index, X_new.y)
purity.idxmax()
X_new.where(X_new.cluster == np.argmax(purity)).dropna(axis = 0).index
cluster_optimum(X, y)


#------------------------------------------------------------------------------
#Implementação do Método Fit() para a proposta melhorada Fixed Sized Levenberg-
#Marquardt (FSLM-LSSVM)
#------------------------------------------------------------------------------
def fit_FSLM_LSSVM_improved(X, y, kappa, mu, Red, N, kernel, tau, epsilon):
    """
    Interface do método
    Ação: Este método visa realizar a poda iterativa dos vetores
    de suporte estimados no LSSVM, trabalhando com a matriz dos
    coeficientes completa em todas as iterações.

    INPUT:
    X: Matriz de features (array N x p);
    y: Vetor de target (array N x 1);
    mu: Taxa de aprendizado (escalar);
    kappa: Termo que define a faixa de poda (escalar);
    Red: Percentual de remoção (escalar);
    N: Número máximo de iterações (escalar);
    kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
    #tau: termo de regularização do problema primal do LSSVM (escalar).
    #epsilon: Tolerância (Critério de parada)

    OUTPUT:
    vetor esparso de multiplicadores de Lagrange estimados;
    Erro Quadrático Médio (MSE) a cada iteração;
    Índices dos Vetores de Suporte para cada iteração.
    """
    
    #Realizando um procedimento de clusterização utilizando o DBSCAN
    #e recuperando os índices do cluster mais ímpuro
    impurity_index = cluster_optimum(X, y)
    
    #Obtendo os índices do complemento do cluster mais ímpuro
    purity_index = list(set(range(X.shape[0])).difference(set(impurity_index)))
    
    #Construção da matriz A e vetor b e inicialização aleatória
    #do vetor de multiplicadores de Lagrange
    A = Construct_A_b(X[purity_index, :], y[purity_index], kernel, tau)['A']
    B = Construct_A_b(X[purity_index, :], y[purity_index], kernel, tau)['B']
    
    A_target = Construct_A_b(X[impurity_index, :], y[impurity_index], kernel, tau)['A']
    B_target = Construct_A_b(X[impurity_index, :], y[impurity_index], kernel, tau)['B']
    
    #Obtenção do número de amostras
    n_samples, n_features = X[purity_index, :].shape
    
    #Inicialização aleatória do vetor de multiplicadores de Lagrange
    z_inicial = np.random.normal(loc=0, scale=1, size=(n_samples + 1, 1))
    z = z_inicial
    
    #Obtenção dos multiplicadores de lagrange para o cluster mais impuro
    z_target = ln.pinv(A_target).dot(B_target)

    #Listas para armazenamento
    idx_suporte = [range(0, n_samples)]
    erros = []
    mult_lagrange = []
    
    #Loop de iteração
    for k in range(1, N + 1):

        #Calculando o erro associado
        erro = B - np.matmul(A, z)
        erro_target = B_target - np.matmul(A_target, z_target)

        #Atualização
        z_anterior = z
        z = z + np.matmul(ln.inv(np.matmul(A.T, A) + mu * np.diag(np.matmul(A.T, A))), np.matmul(A.T, erro))

        #Condição para manter a atualização
        erro_novo = B - np.matmul(A, z)
        if np.mean(erro_novo**2) < np.mean(erro**2):
            z = z
        else:
            mu = mu/10
            z = z_anterior
        
        #Condição para a janela de poda
        if k > kappa and k < N - kappa:
            
            #Realização da poda
            n_colunas_removidas = int((n_samples - (n_samples * Red))/(N - (2 * kappa)))

            #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
            idx_remover = np.argsort(-np.abs(np.squeeze(z)))[:n_colunas_removidas-1]

            #Armazenando os índices dos vetores de suporte a cada iteração
            idx_suporte.append(np.argsort(-np.abs(np.squeeze(z)))[n_colunas_removidas:])
            
            #Adição dos multiplicadores de lagrange
            z_target.append(z[idx_remover])
            
            #Adição das colunas em A_target
            A_target = np.insert(A_target, A_target.shape[1] - 1, A[:, idx_remover])

            #índices dos veotres de suporte
            impurity_index.append(idx_remover)

            #Remoção das colunas de A e linhas de z
            A = np.delete(A, idx_remover, axis = 1)
            z = np.delete(z, idx_remover, axis = 0)
            
        #Outra condição
        if k == N - kappa:

            #Realização da poda
            n_colunas_removidas = int(((n_samples - (n_samples * Red))/(N - (2 * kappa))) + ((n_samples - n_samples * Red)%(N - (2 * kappa))))

            #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
            idx_remover = np.argsort(-np.abs(np.squeeze(z)))[:n_colunas_removidas-1]

            #Armazenando os índices dos vetores de suporte a cada iteração
            idx_suporte.append(np.argsort(-np.abs(np.squeeze(z)))[n_colunas_removidas:])
            
            #Adição dos multiplicadores de lagrange
            z_target.append(z[idx_remover])
            
            #Adição das colunas em A_target
            A_target = np.insert(A_target, A_target.shape[1] - 1, A[:, idx_remover])

            #índices dos veotres de suporte
            impurity_index.append(idx_remover)

            #Remoção das colunas de A e linhas de z
            A = np.delete(A, idx_remover, axis = 1)
            z = np.delete(z, idx_remover, axis = 0)
        
        #Armazenando o erro
        erro_novo_target = B_target - np.matmul(A_target, z_target)
        erros.append(np.mean(erro_target**2))
        
        #Critério de parada
        if np.abs(np.mean(erro_novo_target**2)) < epsilon:
            break
        
    mult_lagrange = np.zeros((X.shape[0]))
    mult_lagrange[impurity_index] = np.squeeze(z_target)[1:]

    #Resultados
    resultados = {"mult_lagrange": mult_lagrange,
                    "b": np.squeeze(z_target)[0],
                    "Erros": erros}

    #Retornando os multiplicadores de Lagrange finais
    return(resultados)

#Realizando alguns testes
resultados = fit_FSLM_LSSVM_improved(X, y, 5, 0.5, 0.2, 30, 'gaussiano', 0.5, 0.01)
resultados['b']
len(resultados['mult_lagrange'])
resultados['mult_lagrange']
for i in resultados['mult_lagrange']:
    print(i)

#Realizando um procedimento de clusterização utilizando o DBSCAN
#e recuperando os índices do cluster mais ímpuro
impurity_index = cluster_optimum(X, y)
impurity_index
len(impurity_index)
X[impurity_index, :]
y[impurity_index]

#Obtendo os índices do complemento do cluster mais ímpuro
purity_index = list(set(range(X.shape[0])).difference(set(impurity_index)))
purity_index
X[purity_index, :]


#Construção da matriz A e vetor b e inicialização aleatória
#do vetor de multiplicadores de Lagrange
A = Construct_A_b(X[purity_index, :], y[purity_index], 'gaussiano', 0.5)['A']
B = Construct_A_b(X[purity_index, :], y[purity_index], 'gaussiano', 0.5)['B']

A_target = Construct_A_b(X[impurity_index, :], y[impurity_index], 'gaussiano', 0.5)['A']
B_target = Construct_A_b(X[impurity_index, :], y[impurity_index], 'gaussiano', 0.5)['B']
    
#Obtenção do número de amostras
n_samples, n_features = X[purity_index, :].shape
    
#Inicialização aleatória do vetor de multiplicadores de Lagrange
z_inicial = np.random.normal(loc=0, scale=1, size=(n_samples + 1, 1))
z = z_inicial
    
#Obtenção dos multiplicadores de lagrange para o cluster mais impuro
z_target = ln.pinv(A_target).dot(B_target)
z_target
#Listas para armazenamento
idx_suporte = [range(0, n_samples)]
erros = []
mult_lagrange = []


for k in range(1, 30 + 1):

        #Calculando o erro associado
        erro = B - np.matmul(A, z)
        erro_target = B_target - np.matmul(A_target, z_target)

        #Atualização
        z_anterior = z
        z = z + np.matmul(ln.inv(np.matmul(A.T, A) + mu * np.diag(np.matmul(A.T, A))), np.matmul(A.T, erro))

        #Condição para manter a atualização
        erro_novo = B - np.matmul(A, z)
        if np.mean(erro_novo**2) < np.mean(erro**2):
            z = z
        else:
            mu = mu/10
            z = z_anterior
        
        #Condição para a janela de poda
        if k > 5 and k < 30 - 5:
            
            #Realização da poda
            n_colunas_removidas = int((n_samples - (n_samples * 0.2))/(30 - (2 * 5)))

            #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
            idx_remover = np.argsort(-np.abs(np.squeeze(z)))[:n_colunas_removidas-1]
