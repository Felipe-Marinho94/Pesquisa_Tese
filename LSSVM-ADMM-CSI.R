#Implementando o LSSVM esparso utilizando o LASSO com
#decomposição do kernel utilizando o algoritmo Cholesky
#Side Information (CSI)
#Data:02/04/2024


#-------------------------------------------------------------------------------
#Carregando algumas bibliotecas relevantes
#-------------------------------------------------------------------------------
library(kernlab)
library(ADMM)
library(ggplot2)
library(plotly)
library(caret)

#-------------------------------------------------------------------------------
#Gerando um dataset sintético
#-------------------------------------------------------------------------------
data(spam)
dt <- as.matrix(spam[c(10:20,3000:3010),-58])

m = 50
n = 100
p = 0.1   #porcentagem de elementos não nulos

x0 = matrix(Matrix::rsparsematrix(n,1,p))
A  = matrix(rnorm(m*n),nrow=m)
for (i in 1:ncol(A)){
  A[,i] = A[,i]/sqrt(sum(A[,i]*A[,i]))
}
b = A%*%x0 + sqrt(0.001)*matrix(rnorm(m))


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
