'''
Implementação do modelo SVM com margem rígida
Autor: Felipe Pinto Marinho
Data:31/08/2023
'''

import numpy as np


class SVM:

    def __init__(self, C = 1.0):
        self.C = C
        self.w = 0
        self.b = 0

    # Função custo
    def hingeloss(self, w, b, x, y):
        # Termo de regularização
        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            # Termo de Otimização
            opt_term = y[i] * ((np.dot(w, x[i])) + b)

            # Cálculo do custo
            loss = reg + self.C * max(0, 1-opt_term)
        return loss[0][0]

    def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=1000):
        # Número de atributos em X
        number_of_features = X.shape[1]

        # Número de atributos em X
        number_of_samples = X.shape[0]

        c = self.C

        # Criando ids de 0 ao number_of_samples - 1
        ids = np.arange(number_of_samples)

        # Embaralhamento das observações
        np.random.shuffle(ids)

        # Criando um array de zeros
        w = np.zeros((1, number_of_features))
        b = 0
        losses = []

        # Gradiente Descendente 
        for i in range(epochs):
            # Calculando o custo
            l = self.hingeloss(w, b, X, Y)

            # Add todos os valores de custo 
            losses.append(l)
            
            # Iniciando de 0 ao número de observações com batch_size como intervalo
            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial+ batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y[x] * (np.dot(w, X[x].T) + b)

                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            # Calculando os gradientes

                            #w.r.t w 
                            gradw += c * Y[x] * X[x]
                            # w.r.t b
                            gradb += c * Y[x]

                # Atualizando os pesos e bias
                w = w - learning_rate * w + learning_rate * gradw
                b = b + learning_rate * gradb
        
        self.w = w
        self.b = b

        return self.w, self.b, losses

    def predict(self, X):
        
        prediction = np.dot(X, self.w[0]) + self.b # w.x + b
        return np.sign(prediction)
    

