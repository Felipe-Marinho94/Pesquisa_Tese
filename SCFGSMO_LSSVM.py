'''
Implemetação do método Spectral Conjugate Functional Gain 
Sequential Minimal Optimization (SCFGSMO) para treinamento
do modelo de máquinas de vetores suporte de mínimos quadrados (LSSVM)
Data:24/08/2024 
'''

#---------------------------------------------------------------
#Carregando alguns pacotes relevantes
#---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kernel import gaussiano_kernel, linear_kernel, polinomial_kernel, construct_K_tilde
from scipy.optimize import line_search
from numpy import linalg, dot

#---------------------------------------------------------------
#Implementando algumas funções relevantes
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

def gradient(alphas, K_tilde, y):
    '''
    Calcula o valor do gradiente da função objetivo
    INPUT:
    alphas - Multiplicadores de Lagrange (array n x 1)
    '''

    return(np.dot(K_tilde, alphas) - y)

def phi(d_anterior, l_anterior):
    '''
    Calcula o parâmetro phi utilzado nas iterações
    '''

    return(dot(d_anterior, l_anterior)/dot(l_anterior, l_anterior))

def rho_otimo(d_anterior, gradiente_anterior, l_anterior, epsilon, p):
    '''
    Calcula o parâmetro rho ótimo utilzado nas iterações
    '''

    return(-dot(d_anterior, gradiente_anterior)/(epsilon * dot(l_anterior, l_anterior) * p))

def phi_barra(d_anterior, l_anterior):
    '''
    Calcula o parâmetro rho barra utilzado nas iterações
    '''

    return(dot(d_anterior, d_anterior)/(dot(d_anterior, l_anterior)))

def p(gradiente, d_anterior, l_anterior):
    '''
    Calcula o parâmetro p utilzado nas iterações
    '''

    return(1- (dot(gradiente, d_anterior)**2)/(dot(gradiente, gradiente)*dot(d_anterior, d_anterior))+
           (dot(gradiente, l_anterior)/(linalg.norm(gradiente)*linalg.norm(l_anterior))+
            linalg.norm(gradiente)/linalg.norm(l_anterior))**2)

def beta_DY(gradiente, d_anterior, l_anterior):
    '''
    Calcula o parâmetro beta DY utilzado nas iterações
    '''

    return(dot(gradiente, gradiente)/dot(d_anterior, l_anterior))


#---------------------------------------------------------------
#Função para ajustar o modelo LSSVM utilizando o método SCFGSMO
#SOURCE: https://journalofinequalitiesandapplications.springeropen.com/articles/10.1186/s13660-020-02375-z
#---------------------------------------------------------------
def fit_SCFGSMO_LSSVM(X, y, gamma, kernel, epsilon, tolerancia, N):
    '''
    Interface do método
    Ação: Este método visa realizar o ajuste do modelo LSSVM empregando
      a metodologia SCFGSMO, retornando os multiplicadores de lagrange
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
    Gradiente = gradient(alphas, K_tilde, y)

    #Direção conjugada inicial
    s = -Gradiente

    #iteração
    k = 0

    #erro
    erro = []

    #Loop de iteração
    while k <= N:

        #Critério de parada
        if linalg.norm(Gradiente) <= tolerancia:
            break

        #Seleção do par de multiplicadores de Lagrange para atualização
        i, j = FGWSS(Gradiente, K_tilde)

        #cálculo do parâmetro rho usando busca linear de wolfe
        rho = -dot(s, Gradiente)/dot(dot(s, K_tilde), s)

        #Atualização
        alphas_anterior = alphas
        Gradiente_anterior = Gradiente
        alphas = alphas + rho * s
        Gradiente = gradient(alphas, K_tilde, y)

        #Armazenando o erro
        erro.append(linalg.norm(Gradiente))

        #calculando os parâmetros theta e beta
        ##Calculando o parâmetro phi
        d_anterior = alphas - alphas_anterior
        l_anterior = Gradiente - Gradiente_anterior

        parameter_phi = phi(d_anterior, l_anterior)
        parameter_phi_barra = phi_barra(d_anterior, l_anterior)
        parameter_p = p(Gradiente, d_anterior, l_anterior)
        parameter_rho_otimo = rho_otimo(d_anterior, Gradiente_anterior, l_anterior, epsilon, parameter_p)
        parameter_beta_DY = beta_DY(Gradiente, d_anterior, l_anterior)

        theta = max(min(parameter_rho_otimo, parameter_phi_barra), parameter_phi)
        beta = theta * parameter_beta_DY

        #Atualização
        s = (-theta * Gradiente) + (beta * d_anterior)

        #Incremento
        k = k + 1
    
    #Resultados
    resultados = {"mult_lagrange": alphas,
                  "erro": erro}
    
    return resultados


X = np.random.normal(10, 1 , size=(1000, 5))
y = np.random.normal(10, 1, 1000)
resultados = fit_SCFGSMO_LSSVM(X, y, 0.5, 'gaussiano', 1.5, 0.01, 100)
resultados['mult_lagrange']
resultados['erro']
len(resultados['erro'])
sns.lineplot(resultados, x=range(0,len(resultados['erro'])), y = resultados['erro'])
plt.xlabel('Iteração')
plt.ylabel('Erro Médio Quadrático (MSE)')
plt.show()




     

