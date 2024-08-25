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
def strong_wolfe(func, grad_func, x, pk, c1=1e-3, c2=0.9,
                 alpha=1.0, alpha_max=100.0, max_iters=100, 
                 verbose=False):
    '''
    Strong Wolfe condition line search method 

    Input:
    func:      the function pointer
    grad_func: the gradien function pointer
    x:         the design variables
    p:         the search direction
    alpha:     the initial estimate for the step length
    alpha_max: the maximum value of alpha 

    returns:
    alpha:     the step length satisfying the strong Wolfe conditions
    '''
    
    # Compute the function and the gradient at alpha = 0
    fk = func(x, linesearch=True)
    gk = grad_func(x)

    # Compute the dot product of the gradient with the search
    # direction to evaluate the derivative of the merit function
    proj_gk = np.dot(gk, pk)

    # Store the old value of the objective
    fj_old = fk
    proj_gj_old = proj_gk
    alpha_old = 0.0

    for j in range(max_iters):
        # Evaluate the merit function
        fj = func(x + alpha*pk, linesearch=True, symb='ro')

        # Evaluate the gradient at the new point
        gj = grad_func(x + alpha*pk)
        proj_gj = np.dot(gj, pk)

        # Check if either the sufficient decrease condition is
        # violated or the objective increased
        if (fj > fk + c1*alpha*proj_gk or
            (j > 0 and fj > fj_old)):
            if verbose:
                print('Sufficient decrease conditions violated: interval found')
            # Zoom and return
            return zoom(func, grad_func, fj_old, proj_gj_old, alpha_old, 
                        fj, proj_gj, alpha,
                        x, fk, gk, pk, c1=c1, c2=c2, verbose=verbose)

        # Check if the strong Wolfe conditions are satisfied
        if np.fabs(proj_gj) <= c2*np.fabs(proj_gk):
            if verbose:
                print('Strong Wolfe alpha found directly')
            func(x + alpha*pk, linesearch=True, symb='go')
            return alpha

        # If the line search is vioalted
        if proj_gj >= 0.0:
            if verbose:
                print('Slope condition violated; interval found')
            return zoom(func, grad_func, fj, proj_gj, alpha, 
                        fj_old, proj_gj_old, alpha_old,
                        x, fk, gk, pk, c1=c1, c2=c2, verbose=verbose)

        # Record the old values of alpha and fj
        fj_old = fj
        proj_gj_old = proj_gj
        alpha_old = alpha

        # Pick a new value for alpha
        alpha = min(2.0*alpha, alpha_max)

        if alpha >= alpha_max:
            if verbose:
                print('Line search failed here')
            return None

    if verbose:
        print('Line search unsuccessful')
    return alpha


def objective(alphas):
    '''
    Calcula o valor da função objetivo
    INPUT:
    alphas - Multiplicadores de Lagrange (array n x 1)
    '''
    
    return(np.dot(np.dot(alphas, K_tilde), alphas) - np.dot(y, alphas))

def gradient(alphas):
    '''
    Calcula o valor do gradiente da função objetivo
    INPUT:
    alphas - Multiplicadores de Lagrange (array n x 1)
    '''

    return(np.dot(K_tilde, alphas) - y)

def d(alpha_posterior, alpha):
    '''
    Calcula a diferença entre os multiplicadores de lagrange
    em iterações sucessivas
    INPUT:
    alpha_posterior - Multiplicadores de Lagrange na iteração
     posterior (array n x 1)
    alpha - multiplicadores de lagrange na iteração atual
    (array n x 1) 
    '''

    return(alpha_posterior - alpha)


def l(gradiente_posterior, gradiente):
    '''
    Calcula a diferença entre os gradientes da função objetivo
    em iterações sucessivas
    INPUT:
    gradiente_posterior - Multiplicadores de Lagrange na iteração
     posterior (array n x 1)
    gradiente - multiplicadores de lagrange na iteração atual
    (array n x 1)
    '''

    return(gradiente_posterior - gradiente)

def phi(d_anterior, l_anterior):
    '''
    Calcula o parâmetro phi utilzado nas iterações
    '''

    return(dot(d_anterior, l_anterior)/dot(l_anterior, l_anterior))

def rho_otimo(s_anterior, gradiente_anterior, l_anterior, epsilon, phi):
    '''
    Calcula o parâmetro rho ótimo utilzado nas iterações
    '''

    return(-dot(s_anterior, gradiente_anterior)/(epsilon * dot(l_anterior, l_anterior) * phi))

def phi_barra(d_anterior, l_anterior):
    '''
    Calcula o parâmetro rho barra utilzado nas iterações
    '''

    return(dot(d_anterior, d_anterior)/(dot(d_anterior, l_anterior)))

def p(gradiente, d_anterior, l_anterior):
    '''
    Calcula o parâmetro p utilzado nas iterações
    '''

    return(1- dot(gradiente, d_anterior)**2/(dot(gradiente, gradiente)*dot(d_anterior, d_anterior))+
           (dot(gradiente, l_anterior)/(linalg.norm(gradiente)*linalg.norm(l_anterior))+
            linalg.norm(gradiente)/linalg.norm(l_anterior))**2)

def beta_DY(gradiente, d_anterior, l_anterior):
    '''
    Calcula o parâmetro beta DY utilzado nas iterações
    '''

    return(dot(gradiente, gradiente)/dot(d_anterior, l_anterior))


K_tilde = np.random.normal(loc=0, scale=1, size=(100, 100))
alphas_start = np.random.normal(loc=0, scale=1, size=100)
direction = np.random.normal(loc=0, scale=1, size=100)
strong_wolfe(objective, gradient, alphas_start, direction)


#---------------------------------------------------------------
#Função para ajustar o modelo LSSVM utilizando o método SCFGSMO
#SOURCE: https://journalofinequalitiesandapplications.springeropen.com/articles/10.1186/s13660-020-02375-z
#---------------------------------------------------------------
def fit_SCFGSMO_LSSVM(X, y, K_tilde, epsilon, N):
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

    n_samples, n_features = X.shape
    

    #Inicialização
    #Multiplicadores de Lagrange
    alphas = np.random.normal(loc=0, scale=1, size=n_samples)

    #Gradiente
    Gradiente = gradient(alphas)

    #Objetivo
    objetivo = objective(alphas)

    #Direção conjugada inicial
    s = -Gradiente

    #iteração
    k = 0

    #erro
    erro = []

    #Loop de iteração
    while k <= N:

        #Critério de parada
        if linalg.norm(Gradiente) <= epsilon:
            break

        #cálculo do parâmetro rho usando busca linear de wolfe
        rho = line_search(objective, gradient, alphas, s)[0]

        #Atualização
        alphas_anterior = alphas
        Gradiente_anterior = Gradiente
        alphas =+ rho * s
        Gradiente = gradient(alphas)

        #Armazenando o erro
        erro.append(linalg.norm(Gradiente))

        #calculando os parâmetros theta e beta
        ##Calculando o parâmetro phi
        d_anterior = d(alphas, alphas_anterior)
        l_anterior = l(Gradiente, Gradiente_anterior)

        parameter_phi = phi(d_anterior, l_anterior)
        parameter_phi_barra = phi_barra(d_anterior, l_anterior)
        parameter_p = p(Gradiente, s, l_anterior)
        parameter_rho_otimo = rho_otimo(s, Gradiente_anterior, l_anterior, epsilon, parameter_p)
        parameter_beta_DY = beta_DY(Gradiente, d_anterior, l_anterior)

        theta = max(min(parameter_rho_otimo, parameter_phi_barra), parameter_phi)
        beta = theta * parameter_beta_DY

        #Atualização
        s = -theta * Gradiente + beta * s

        #Incremento
        k =+ 1
    
    #Resultados
    resultados = {"mult_lagrange": alphas,
                  "erro": erro}
    
    return resultados


X = np.random.normal(10, 1 , size=(1000, 5))
y = np.random.normal(10, 1, 1000)
K_tilde = construct_K_tilde(X, 2, "gaussiano")

resultados = fit_SCFGSMO_LSSVM(X, y, K_tilde, 0.01, 100)
len(resultados['mult_lagrange'])
resultados['erro']
sns.lineplot(resultados, x=range(0,len(resultados['erro'])), y = resultados['erro'])
plt.xlabel('Iteração')
plt.ylabel('Erro Médio Quadrático (MSE)')
plt.show()




     

