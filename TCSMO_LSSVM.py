'''
Implementação da proposta fast LSSVM Vvia Three term conjugate-like SMO algorithm
Data:27/08/2024
'''

#-------------------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#-------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kernel import linear_kernel, polinomial_kernel, gaussiano_kernel

#---------------------------------------------------------------
#Implementação da estratégia de seleção do conjunto de trabalho
#Baseado no máximo ganho funcional
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

#--------------------------------------------------------------------------------
#Implementação do método fit() para ajustar o LSSVM utilizando
#uma variante do algoritmo SMO utilizando direções conjugadas de
#três termos
#SOURCE:https://www.sciencedirect.com/science/article/abs/pii/S0031320323001784
#--------------------------------------------------------------------------------
def fit_TCSMO_LSSVM(X, y, gamma, kernel, epsilon, N):
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
    z_penultimo = np.zeros(n_samples)
    z_ultimo = np.zeros(n_samples)

    #parâmetros nu
    nu_penultimo = np.zeros(n_samples)
    nu_ultimo = np.zeros(n_samples)

    #termo tau
    tau_penultimo = 1
    tau_ultimo = 1

    #Controle de iteração
    k = 0

    #erro
    erro = []

    while k <= N:
        
        #Seleção do conjunto de trabalho
        i, j = FGWSS(Gradient, K_tilde)

        #Cálculo do parâmetro conjugado
        delta_penultimo = (nu_penultimo[j] - nu_penultimo[i])/tau_penultimo
        delta_ultimo = (nu_ultimo[j] - nu_ultimo[i])/tau_ultimo

        #Construindo a direção descendente conjugada
        z = np.eye(1, n_samples, i)[0] - np.eye(1, n_samples, j)[0] + delta_penultimo * z_penultimo + delta_ultimo * z_ultimo

        #Atualização dos z
        z_ultimo = z_penultimo
        z_penultimo = z

        #Atualização dos vetores auxiliares
        nu = K_tilde[:, i] - K_tilde[:, j] + delta_penultimo * nu_penultimo + delta_ultimo * nu_ultimo
        nu_ultimo = nu_penultimo
        nu_penultimo = nu

        #Calculando o tau
        tau = nu[i] - nu[j]

        #Atualização dos taus
        tau_ultimo = tau_penultimo
        tau_penultimo = tau

        #Calculando o passo utilizando line search
        rho = (Gradient[j] - Gradient[i])/tau

        #Atualização dos multiplicadores de Lagrange
        alphas = alphas + rho * z

        #Atualização do gradiente
        Gradient = Gradient + rho * nu

        #Armazenando o erro
        erro.append(np.max(Gradient) - np.min(Gradient))

        #Condição de parada
        if np.abs(np.max(Gradient) - np.min(Gradient)) <= epsilon:
            break
        
        #Resultado
        resultados = {"mult_lagrange": alphas,
                      "erro": erro}
    
    return resultados


X = np.random.normal(10, 1 , size=(1000, 5))
y = np.random.normal(10, 1, 1000)

resultados = fit_TCSMO_LSSVM(X, y, 0.5, 'gaussiano', 0.01, 500)
len(resultados['mult_lagrange'])
resultados['erro']
sns.lineplot(resultados, x=range(0,len(resultados['erro'])), y = resultados['erro'])
plt.xlabel('Iteração')
plt.ylabel('Erro Médio Quadrático (MSE)')
plt.show()





