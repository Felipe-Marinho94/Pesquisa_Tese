#Implementação de abordagens clássicas para a solução do SVM
#Gradiente Descendente (Primal), Programa Quadrático (Dual), SMO (DUAL)
#Autor: Felipe Pinto Marinho
#Data: 24/12/2023

#Carregando alguns pacotes relevantes
library(MASS)
library(ggplot2)
library(plotly)


#Implementando algumas funções relevantes
indicador = function(condition) ifelse(condition, 1, -1)

#Teste da função
a = 3
indicador(a < 1)


predict_SVM = function(w_opt, X){
  
  #A matriz X deve incluir o bias
  #função para calcular a saída
  y = X %*% w_opt
  return(indicador(y > 0))
  
}

#Teste da função
predict_SVM(w_answer, X)

#Geração de malha
make.grid = function(x, n = 75) {
  grange = apply(x, 2, range)
  x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
  x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
  expand.grid(X1 = x1, X2 = x2)
}

#Teste da função
xgrid = make.grid(x)
xgrid[1:10,]

#gradiente descendente
svm_gradient = function(x, eta=0.001, R=10000){
  X = cbind(1,x) #Matriz de atributos com bias
  n = nrow(X)  #número de observações
  p = ncol(X) ##número de atributos mais bias
  w_intial = rep(0,p) #Chute inicial
  W = matrix(w_intial ,nrow = R+1,ncol = p,byrow = T) #Matriz chute inicial
  for(i in 1:R){
    for(j in 1:p)
    {
      W[i+1,j] = W[i,j]+eta*sum(((y*(X%*%W[i,]))<1)*1 * y * X[,j] )  
    }
  }
  return(W)  
}

get_svm = function(x){
  w_answer = svm_gradient(x)[nrow(svm_gradient(x)),]
  return(w_answer )
}

#Criação de um dataset sintético linearmente separável
n = 30
a1 = rnorm(n)
a2 = 1 - a1 + 2* runif(n)
b1 = rnorm(n)
b2 = -1 - b1 - 2*runif(n)
x = rbind(matrix(cbind(a1,a2),,2),matrix(cbind(b1,b2),,2))
y = matrix(c(rep(1,n),rep(-1,n)))
plot(x,col=ifelse(y>0,4,2),pch=".",cex=7,xlab = "x1",ylab = "x2")
w_answer = get_svm(x)
abline(-w_answer[1]/w_answer[3],-w_answer[2]/w_answer[3])
abline((1-w_answer[1])/w_answer[3],-w_answer[2]/w_answer[3],lty=2)
abline((-1-w_answer[1])/w_answer[3],-w_answer[2]/w_answer[3],lty=2)

#Fronteira de decisão
xgrid = make.grid(x)
X_grid = as.matrix(cbind(1, xgrid))
ygrid = predict_SVM(w_answer, X_grid)
plot(xgrid, col = ifelse(ygrid>0, 4, 2), pch = 20, cex = .2, main = "Decision Boundary")
points(x, col = ifelse(y>0, 4, 2), pch = ifelse(y>0, 19, 17))
points(support_vectors, pch = 5, cex = 2)
legend("topright", legend=c("Output 1", "Output -1"),
       col=c("blue", "red"), pch = c(19, 17), cex=0.8,
       title="Output class", text.font=4, bg='lightblue')

abline(-w_answer[1]/w_answer[3],-w_answer[2]/w_answer[3])
abline((1-w_answer[1])/w_answer[3],-w_answer[2]/w_answer[3],lty=2)
abline((-1-w_answer[1])/w_answer[3],-w_answer[2]/w_answer[3],lty=2)


