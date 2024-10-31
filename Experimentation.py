'''
Script para realização das etapas de carregamento de dataset,
divisão treino/validação/teste, tuning dos modelos e hold-out
para obtenção das métricas de performance
Data: 29/10/2024
'''

#--------------------------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#--------------------------------------------------------------------------------------------------
#Manipulação de dados
import pandas as pd
import numpy as np

#Visualização gráfica
import matplotlib.pyplot as plt
import seaborn as sns

#DS e Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Propostas
from LSSVM_ADMM import fit_LSSVM_ADMM, predict_class_LSSVM_ADMM #Primeira proposta
from FSLM_LSSVM import fit_FSLM_LSSVM #proposta do professor
from FSLM_LSSVM_improved import fit_FSLM_LSSVM_improved #Segunda proposta
from SCG_LSSVM import fit_SCG_LSSVM #Terceira proposta
from TCSMO_LSSVM import fit_TCSMO_LSSVM #Quarta proposta

#Modelos para comparação
from P_LSSVM import fit_P_LSSVM
from IP_LSSVM import fit_IP_LSSVM
from CFGSMO_LSSVM import fit_CFGSMO_LSSVM
from kernel import gaussiano_kernel, fit_class, predict_class #LSSVM clássico e pré-condicionado
from svm_tese import SVM #SVM resolvido por QP

#Tempo de processamento
from time import process_time

#Performance
from métricas import metricas

#Outros
import os
import pathlib

#--------------------------------------------------------------------------------------------------
#Implementando algumas funções relevantes
#--------------------------------------------------------------------------------------------------
#Carregando dataset
def load_data(data_path, name_dataset):
    '''
    Função para carregar os datasets em um pandas
    Dataframe, automatizando o processo de obtenção
    das performances dos modelos
    INPUT:
    data_path: Caminho indicando o diretório do dataset;
    name_dataset: Nome do dataset (string)

    OUTPUT:
    pandas Dataframe com os dados considerados. 
    '''
    extension = pathlib.Path(name_dataset).suffix
    name_dataset = pathlib.Path(name_dataset).stem
    csv_path = os.path.join(data_path, f"{name_dataset}{extension}")
    return pd.read_csv(csv_path)

#Inicialização do caminho para os arquivos csv's
data_path = '/Users/Felipe/Documents/Python Scripts/Tese/Datasets'

#Realização de um pequeno teste
df = load_data(data_path, 'banana.csv')

#Remoção de registros com valores nulos
def drop_nan(dataset):
    '''
    Remove registros (linhas) com valores nulos
    INPUT:
    dataset: Dataframe com os dados de entrada (Dataframe);

    OUTPUT:
    Dataframe sem as linhas com valores nulos.
    '''

    return dataset.dropna(axis = 0)

#--------------------------------------------------------------------------------------------------
#Implementando o pipeline de dados
#-------------------------------------------------------------------------------------------------- 
#Inicialização dos nomes dos datasets
datasets = ['banana.csv', 'breast-cancer-wisconsin.data', 'haberman.data',
'hepatitis.data', 'ionosphere.data', 'column_2C.dat', 'SouthGermanCredit.asc']

#Definição de um pipeline de dados
data_pipeline = Pipeline([('drop_nan', drop_nan()),
                          ('std_scaler', StandardScaler())])

for dataset in datasets:

    #Carregando a base
    df = load_data(data_path, dataset)

    #Separando features de target
    X = df.iloc[:, : (X.shape[1]-1)]
    y = df.iloc[:, X.shape[1]]

    #Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)





