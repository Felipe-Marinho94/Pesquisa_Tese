'''
Implemetação do método Conjugate Functional Gain 
Sequential Minimal Optimization (CFGSMO) para treinamento
do modelo de máquinas de vetores suporte de mínimos quadrados (LSSVM)
Data:12/08/2024 
'''

#---------------------------------------------------------------
#Carregando alguns pacotes relevantes
#---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kernel import gaussiano_kernel, linear_kernel, polinomial_kernel

#---------------------------------------------------------------
#Implementação de alguma funções relevantes
#---------------------------------------------------------------
def FGWSS(gradient, K):
    '''
    Interface do Método
    Ação: Visa selecionar os índices dos dois multiplicadores de
    Lagrange que serão atualizados, por meio da heurística do
    método Functional Gain Working Selection Stategy (FGWSS)

    INPUT:
    gradient: Vetor Gradiente (array n)
    K: Matriz de Kernel (array n x n)

    OUTPUT:
    os dois índices selecionados
    '''
    
    #Primeiro índice
    i = np.argmax(np.absolute(gradient))

    #Segundo índice
    exception = i
    m = np.zeros(gradient.shape[0], dtype=bool)
    m[exception] = True

    numerador = (gradient-gradient[i])**2
    denominador = 2 * (K[i, i] * np.ones(K.shape[0]) + np.diag(K) - K[i, :] - K[:, i])
    quociente = np.ma.array(numerador/denominador, mask=m)
    j = np.argmax(quociente)

    return (i, j)



#---------------------------------------------------------------
#Função para ajustar o modelo LSSVM utilizando o método CFGSMO
#SOURCE: https://link.springer.com/article/10.1007/s00521-022-07875-1
#---------------------------------------------------------------
def fit_CFGSMO_LSSVM(X, y, gamma, kernel, epsilon, N):
    '''
    Interface do método
    Ação: Este método visa realizar o ajuste do modelo LSSVM empregando
      a metodologia CFGSMO, retornando os multiplicadores de lagrange
      ótimos para o problema de otimização.

    INPUT:
    X: Matriz de features (array N x p);
    y: Vetor de target (array N x 1);
    gamma: termo de regularização de Tichonov (escalar)
    kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
    epsilon: Tolerância (Critério de parada)
    N: Número máximo de iterações

    OUTPUT:
    vetor ótimo de multiplicadores de Lagrange estimados.
    '''

    #Construindo a matriz de kernel
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
            
    
    #Regularização de Tichonov
    K_tilde = K + (1/gamma)*np.diag(np.full(K.shape[0], 1))
    
    #Inicialização
    #Multiplicadores de Lagrange
    alphas = np.zeros(n_samples)

    #Gradiente
    Gradient = -y

    #direções conjugadas
    s = np.zeros(n_samples)
    t = np.zeros(n_samples)

    #Termo tau
    tau = 1

    #Controle de iteração
    k = 0

    #erro
    erro = []

    #Laço de iteração
    while k <= N:

        #incremento
        k = k + 1

        #Seleção do par de multiplicadores de Lagrange para atualização
        i, j = FGWSS(Gradient, K_tilde)

        #Realizando as atualizações
        r = (t[j] - t[i])/tau
        s = np.eye(1, n_samples, i)[0] - np.eye(1, n_samples, j)[0] + r * s
        t = K_tilde[:, i] - K_tilde[:, j] + r * t
        tau = t[i] - t[j]

        #Calculando o parâmetro rho
        rho = (Gradient[j] - Gradient[i])/tau

        #Atualização da variável Dual
        alphas = alphas + rho * s

        #Atualização do gradiente
        Gradient = Gradient + rho * t

        #Armazenando o erro
        erro.append(np.max(Gradient) - np.min(Gradient))

        #Condição de parada
        if np.abs(np.max(Gradient) - np.min(Gradient)) <= epsilon:
            break
    
    #Resultados
    resultados = {"mult_lagrange": alphas,
                  "erro": erro}
    
    return resultados
    

        
#Realizando alguns testes
K = np.random.normal(loc=0, scale=1, size=(100, 100))
A = 2 * (K[1, 1] * np.ones(K.shape[0]) + np.diag(K) - K[1, :] - K[:, 1])
G = np.random.normal(loc=0, scale=1, size=(100))
np.argmax(G/A)

X = np.random.normal(10, 1 , size=(1000, 5))
y = np.random.normal(10, 1, 1000)

resultados = fit_CFGSMO_LSSVM(X, y, 100, 'gaussiano', 0.01, 500)
len(resultados['mult_lagrange'])
resultados['erro']
sns.lineplot(resultados, x=range(0,len(resultados['erro'])), y = resultados['erro'])
plt.xlabel('Iteração')
plt.ylabel('Erro Médio Quadrático (MSE)')
plt.show()

10* np.array([1, 2, 3])