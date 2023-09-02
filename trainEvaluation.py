from sklearn import datasets
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from SVM import SVM

# Criando o dataset
'''
X, y = datasets.make_blobs(

        n_samples = 100, # Número de observações
        n_features = 2, # Atributos
        centers = 2,
        cluster_std = 1,
        random_state=40
    )

# Classes 1 e -1
y = np.where(y == 0, -1, 1)
'''

#Importando um dataset de classificação binária
bd = pd.read_csv('wdbc.data', sep=',', header=0)

#Pré-processamento
bd.iloc[:, 1] = bd.iloc[:, 1].replace(['M', 'B'], [-1, 1])
print(bd.iloc[:, 1])
X = bd.drop(bd.columns[[0, 1]], axis=1)
print(X.head())
y = bd.iloc[:, 1]
print(y.head())

#Normalização estatística padrão
scale = StandardScaler()
X_processed = scale.fit_transform(X) 
print(X_processed[0])
y_array = y.to_numpy()
print(y_array)

#Dividindo o dataset
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_array, test_size=0.5, random_state=42)

#Treinando o modelo SVM
svm = SVM()

w, b, losses = svm.fit(X_train, y_train)

#Realizando predições e avaliando a acurácia de teste
prediction = svm.predict(X_test)

# Valor Loss
lss = losses.pop()

print("Loss:", lss)
print("Prediction:", prediction)
print("Accuracy:", accuracy_score(prediction, y_test))
print("w, b:", [w, b])

#Visualização dos resultados
def visualize_dataset():
    plt.scatter(X_processed[:, 0], X_processed[:, 1], c=y_array)


# Visualizando SVM
def visualize_svm():

    def get_hyperplane_value(x, w, b, offset):
        return (-w[0][0] * x + b + offset) / w[0][1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=y_test)

    x0_1 = np.amin(X_test[:, 0])
    x0_2 = np.amax(X_test[:, 0])

    x1_1 = get_hyperplane_value(x0_1, w, b, 0)
    x1_2 = get_hyperplane_value(x0_2, w, b, 0)

    x1_1_m = get_hyperplane_value(x0_1, w, b, -1)
    x1_2_m = get_hyperplane_value(x0_2, w, b, -1)

    x1_1_p = get_hyperplane_value(x0_1, w, b, 1)
    x1_2_p = get_hyperplane_value(x0_2, w, b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X_processed[:, 1])
    x1_max = np.amax(X_processed[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()


visualize_dataset()
visualize_svm()
