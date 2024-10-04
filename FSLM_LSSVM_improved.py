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
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from matplotlib import style

#------------------------------------------------------------------------------
#Implementação de algumas funções relevantes
#------------------------------------------------------------------------------
def Construct_A_B(X, y, kernel, tau):
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
    
    #--------------------------------------------------------------------------
    #Decomposição da matriz de kernel
    #--------------------------------------------------------------------------
    #Cholesky Incompleta
    P = np.linalg.cholesky(K + 0.01 * np.diag(np.full(K.shape[0], 1)))
    
    #Construção da matriz dos coeficiente A
    A = P.T
    
    #Construção do vetor de coeficientes b
    B = np.dot(ln.inv(tau * np.identity(n_samples) + np.dot(P, P.T)), np.dot(P.T, y))
    B = np.expand_dims(B, axis = 1)
    
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

Construct_A_B(X, y, 'gaussiano', 0.5)['A'].shape
Construct_A_B(X, y, 'gaussiano', 0.5)['B'].shape

###############################################################################
def purity_level(X, y):
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
purity_level(X, y)

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
    purity = df.groupby('cluster').apply(purity_level, df.y)
    
    return(df.where(df.cluster == purity.idxmax()).dropna(axis = 0).index)


#Realizando alguns testes
clustering = DBSCAN().fit(pd.DataFrame(X))
cluster = pd.Series(clustering.labels_, name = "cluster")    
X_new = pd.concat([cluster, pd.DataFrame(X), pd.Series(y, name = "y")],
                  axis = 1)
X_new.head()
purity = X_new.groupby('cluster', group_keys = True).apply(purity_level, X_new.y)
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
    
    #Obtenção do B para todo o dataset
    B_total = Construct_A_B(X, y, kernel, tau)['B']
    
    #Realizando um procedimento de clusterização utilizando o DBSCAN
    #e recuperando os índices do cluster mais ímpuro
    impurity_index = cluster_optimum(X, y)
    
    #Obtendo os índices do complemento do cluster mais ímpuro
    purity_index = np.array(list(set(range(X.shape[0])).difference(set(impurity_index))))
    purity_index = np.squeeze(purity_index)
    
    #Construção da matriz A e vetor b e inicialização aleatória
    #do vetor de multiplicadores de Lagrange
    A = Construct_A_B(X[purity_index, :], y[purity_index], kernel, tau)['A']
    B = Construct_A_B(X[purity_index, :], y[purity_index], kernel, tau)['B']
    
    A_impurity = Construct_A_B(X[impurity_index, :], y[impurity_index], kernel, tau)['A']
    B_impurity = Construct_A_B(X[impurity_index, :], y[impurity_index], kernel, tau)['B']
    
    #Obtenção do número de amostras
    n_samples, n_features = X[purity_index, :].shape
     
    #Inicialização aleatória do vetor de multiplicadores de Lagrange
    z_inicial = np.random.normal(loc=0, scale=1, size=(n_samples))
    z = pd.Series(z_inicial, index=purity_index)
    A = pd.DataFrame(A, index = purity_index, columns = purity_index)
    
    #Obtenção dos multiplicadores de lagrange para o cluster mais impuro
    z_target = np.zeros(X.shape[0])
    z_impurity = ln.pinv(A_impurity).dot(B_impurity)
    z_target[impurity_index] = np.squeeze(z_impurity)
    
    #Obtenção do A_target
    A_target = np.zeros((X.shape[0], X.shape[0]))
    A_target[np.ix_((impurity_index), (impurity_index))] = A_impurity

    #Listas para armazenamento
    erros = []
    idx_suporte = []
    
    #Loop de iteração
    for k in range(1, N + 1):

        #Calculando o erro associado
        erro = np.squeeze(B) - np.matmul(A, z)
        erro_target = np.squeeze(B_total) - np.matmul(A_target, z_target)

        #Atualização
        z_anterior = z
        z = z + np.matmul(ln.inv(np.matmul(A.T, A) + mu * np.diag(np.matmul(A.T, A))), np.matmul(A.T, erro))

        #Condição para manter a atualização
        erro_novo = np.squeeze(B) - np.matmul(A, z)
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
            #idx_remover = np.argsort(-np.abs(np.squeeze(z)))[:n_colunas_removidas-1]
            sorted_indices = np.abs(z).sort_values(ascending = False).index
            idx_remover = sorted_indices[:n_colunas_removidas]

            #Adição dos multiplicadores de lagrange
            z_target[idx_remover] = z[idx_remover]
            
            #Adição das colunas em A_target
            A_target[np.ix_((purity_index), (idx_remover))] = A.loc[purity_index, idx_remover]

            #índices dos vetores de suporte
            impurity_index = np.append(impurity_index, (idx_remover))
            idx_suporte.append(impurity_index)
            
            #Remoção das colunas de A e linhas de z
            A = A.drop(idx_remover, axis = 1)
            #z = np.delete(z, idx_remover, axis = 0)
            z = z.drop(idx_remover)
            
            #Atualização dos indices puros
            purity_index = z.index

        #Outra condição
        if k == N - kappa:

            #Realização da poda
            n_colunas_removidas = int(((n_samples - (n_samples * Red))/(N - (2 * kappa))) + ((n_samples - n_samples * Red)%(N - (2 * kappa))))

            #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
            #idx_remover = np.argsort(-np.abs(np.squeeze(z)))[:n_colunas_removidas-1]
            sorted_indices = np.abs(z).sort_values(ascending = False).index
            idx_remover = sorted_indices[:n_colunas_removidas]

            #Adição dos multiplicadores de lagrange
            z_target[idx_remover] = z[idx_remover]
            
            #Adição das colunas em A_target
            A_target[np.ix_((purity_index), (idx_remover))] = A.loc[purity_index, idx_remover]

            #índices dos vetores de suporte
            impurity_index = np.append(impurity_index, (idx_remover))
            idx_suporte.append(impurity_index)
            
            #Remoção das colunas de A e linhas de z
            A = A.drop(idx_remover, axis = 1)
            #z = np.delete(z, idx_remover, axis = 0)
            z = z.drop(idx_remover)
            
            #Atualização dos indices puros
            purity_index = z.index
        
        #Armazenando o erro
        erro_novo_target = np.squeeze(B_total) - np.matmul(A_target, z_target)
        erros.append(np.mean(erro_target**2))
        
        #Critério de parada
        if np.abs(np.mean(erro_novo_target**2)) < epsilon:
            break
        
    mult_lagrange = z_target

    #Resultados
    resultados = {"mult_lagrange": mult_lagrange,
                    "Erros": erros,
                    "Indices_multiplicadores": idx_suporte,
                    "Iteração": k}

    #Retornando os multiplicadores de Lagrange finais
    return(resultados)

#Realizando alguns testes
resultados = fit_FSLM_LSSVM_improved(X, y, 5, 0.5, 0.2, 30, 'gaussiano', 0.5, 0.01)
len(resultados['mult_lagrange'])
resultados['mult_lagrange']
resultados['Erros']
resultados['Iteração']
len(resultados['Indices_multiplicadores'])
for i in resultados['mult_lagrange']:
    print(i)


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
fig.scatter(x=X[:,0], y=X[:,1])
fig.plot(X[index,0], X[index,1], "or")

index = resultados['Indices_multiplicadores'][5]
index.shape
fig1 = plt.subplot(2, 2, 2)
fig1.scatter(x=X[:,0], y=X[:,1])
fig1.plot(X[index,0], X[index,1], "or")

index = -resultados['Indices_multiplicadores'][10]
index.shape
fig2 = plt.subplot(2, 2, 3)
fig2.scatter(x=X[:,0], y=X[:,1])
fig2.plot(X[index,0], X[index,1], "or")

index = -resultados['Indices_multiplicadores'][16]
index.shape
fig3 = plt.subplot(2, 2, 4)
fig3.scatter(x=X[:,0], y=X[:,1])
fig3.plot(X[index,0], X[index,1], "or")

plt.show()

c = np.array([-1, -2, -3])
c = pd.Series(c)
np.abs(c)
np.abs(c).sort_values(ascending = False).index