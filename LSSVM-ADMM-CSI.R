#Implementando o LSSVM esparso utilizando o LASSO com
#decomposição do kernel utilizando o algoritmo Cholesky
#Side Information (CSI)
#Data:02/04/2024


#-------------------------------------------------------------------------------
#Carregando algumas bibliotecas relevantes
#-------------------------------------------------------------------------------
library(kernlab)
library(ADMM)
library(caret)
library(pracma)
library(ISLR)
library(dplyr)

#-------------------------------------------------------------------------------
#Implementando algumas funções relevantes
#-------------------------------------------------------------------------------
linear_kernel = function(x, y){
  return(dot(x,y))
}

polinomial_kernel = function(x, y, C = 1, d = 3){
  return((dot(x, y) + C)^{d})
}

gaussiano_kernel = function(x, y, gamma = 0.1){
  return(exp(-gamma * norm(x - y)^(2)))
}

#-------------------------------------------------------------------------------
#Gerando um método para fit() do modelo proposto
#-------------------------------------------------------------------------------
fit_LSSVM_ADMM = function(X, y, tau){
  #Input
  #X: Matriz de Dados (array n x p)
  #y: Vetor de rótulos (classificação) 
  #   ou vetor de respostas numéricas (regressão) (array n x 1)
  #kernel: kernel utlizado ("linear", "polinomial", "gaussiano") (string)
  #tau: Termo de regularização do problema primal do LSSVM (escalar)
  #Output
  #x_ótimo: Vetor solução aproximado do sistema KKT Ax = b (array n+1 x 1)
  #x_ótimo = [alphas b].T
  
  #Obtenção da matriz de kernel
  rbf = rbfdot(sigma = 0.5) #Instanciando o kernel gaussiano
  K = kernelMatrix(rbf, X)
  
  #Decomposição da matriz de kernel utilizando o algoritmo
  #Cholesky Side Information (CSI)
  #source: http://cmm.ensmp.fr/~bach/bach_jordan_csi.pdf
  P = csi(as.matrix(X), y, kernel = "rbfdot", kpar = list(sigma = 0.5), rank = 3)
  
  #Construção da matriz dos coeficientes A
  A = t(P)
  
  #Construção do vetor de termos indenpendentes b
  b = inv(tau * diag(nrow(X)) + crossprod(t(P))) %*% crossprod(P, y)
  
  #Configurando o valor de lambda
  lambda = 0.1*base::norm(t(A)%*%b, "F")
  
  #Solução do sistema KKT em conjunto com o LASSO
  solution  = admm.lasso(A, b, lambda)$x
  
  #Retornando os multiplicadores de Lagrange ótimos em conjunto
  #com a matriz de kernel
  resultados = list(solution, K)
  names(resultados) = c("alphas", "kernel")
  return(resultados)
}


#-------------------------------------------------------------------------------
#Gerando um método para predict() do modelo proposto
#-------------------------------------------------------------------------------
predict_LSSVM_ADMM = function(alphas, b, kernel, X_treino, y_treino, X_teste){
  
  #Inicialização
  estimado = rep(0, nrow(X_teste))
  n_treino = nrow(X_treino)
  n_teste = nrow(X_teste)
  K = matrix(0, nrow = n_teste, ncol = n_treino)
  
  #Construção da matriz de kernel
  for (i in 1:n_teste) {
    for (j in 1:n_treino) {
      
      if (kernel == "linear"){
        K[i, j] = linear_kernel(X_teste[i, ], X_treino[j, ])
      }
      
      if (kernel == "polinomial"){
        K[i, j] = polinomial_kernel(X_teste[i, ], X_treino[j, ])
      }
      
      if (kernel == "gaussiano"){
        K[i, j] = gaussiano_kernel(X_teste[i, ], X_treino[j, ])
      }
    }
    
    #Realização da predição
    estimado[i] = sign(sum((alphas * y_treino) * K[i, ]) + b)
  }
  
}

#-------------------------------------------------------------------------------
#Testando o modelo proposto no Breast Cancer dataset
#-------------------------------------------------------------------------------
df = Default

#Convertendo a variável de saída
df$default = ifelse(df$default == "Yes", -1, 1)
table(df$default)

#Separação treino/teste
treino = sample(nrow(df), 0.7*nrow(df))
names(df)
X_treino = select_if(df[treino, -1], is.numeric)
X_teste = select_if(df[treino, -1], is.numeric)
y_treino = df$default[treino]
y_teste = df$default[-treino]

#Ajuste do modelo proposto
resultados_LSSVM_ADMM = fit_LSSVM_ADMM(X_treino, y_treino, 0.5)

#-------------------------------------------------------------------------------
#Decomposição de posto reduzido da matriz de kernel, da forma K = GG'
#Usando o algoritmo CSI
#source: http://cmm.ensmp.fr/~bach/bach_jordan_csi.pdf
#-------------------------------------------------------------------------------
#Geração da matriz de Gram para a matriz de dados A
rbf = rbfdot(sigma = 0.5) #Instanciando o kernel gaussiano
kernel = kernelMatrix(rbf, A)
fix(kernel)

#Decomposição da matriz de Gram utilizando o algoritmo CSI
G = csi(A, b, kernel = "rbfdot", kpar = list(sigma = 0.5), rank = 3)
K_hat = crossprod(t(G))
norm(kernel - K_hat, type = "F")
kernel - K_hat

#-------------------------------------------------------------------------------
#Aplicação do ADMM para a solução do LASSO utilizando o problema primal do
#Least Square Support Vector Machine
#source:https://ieeexplore.ieee.org/document/7878713
#source:https://stanford.edu/~boyd/papers/admm_distr_stats.html
#-------------------------------------------------------------------------------
## set regularization lambda value
lambda = 0.1*base::norm(t(A)%*%b, "F")

## run example
output  = admm.lasso(A, b, lambda)
niter   = length(output$history$s_norm)
history = output$history

## report convergence plot
opar <- par(no.readonly=TRUE)
par(mfrow=c(1,3))
plot(1:niter, history$objval, "b", main="cost function")
plot(1:niter, history$r_norm, "b", main="primal residual")
plot(1:niter, history$s_norm, "b", main="dual residual")
par(opar)
output$x
