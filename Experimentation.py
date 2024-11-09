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
from sklearn.preprocessing import StandardScaler, FunctionTransformer

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
from kernel import gaussiano_kernel, fit_LSSVM, predict_class #LSSVM clássico e pré-condicionado
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

#Análise Univariada
def filtro_volumetria(df):
    '''
    função para remover features com um percentual de nulos
    acima do valor do threshold
    INPUT:
        df - Pandas dataframe com as features a serem analisadas;
        threshold - Limiar utilizado como critério para a remoção
        da feature
    
    OUTPUT:
        pandas dataframe somente com as features com boa volumetria
    '''
    
    for column in df.columns:
        if df[column].isna().sum()/df.shape[0] > 0.8:
            df = df.drop([column], axis = 1)
    
    return df

def filtro_volatilidade(df):
    '''
    função para remover features com baixa variabilidade, que por
    sua vez, não contribuem significativamente para a discriminação
    interclasses
    INPUT:
        df - Pandas dataframe com as features a serem analisadas;
        threshold - Limiar utilizado como critério para a remoção
        da feature
    
    OUTPUT:
        pandas dataframe somente com as features com boa volatilidade
    '''
    
    for column in df.select_dtypes(include = ["int64", "float64"]).columns.values:
        if round(df[column].var(), 2) < 0.9:
            df = df.drop([column], axis = 1)
    return(df)

def filtro_correlacao(df):
    '''
    função para remover features colineares com base na correlação de spearmen
    INPUT:
        df - Pandas dataframe com as features a serem analisadas;
        threshold - Limiar utilizado como critério para a remoção
        da feature
    
    OUTPUT:
        pandas dataframe somente com as features descorrelacionadas
    '''

    # Calcula a matriz de correlação de spearmen
    corr_matrix = df.corr(method = 'spearman')
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            if val >= 0.7:
    
                drop_cols.append(col.values[0])

    
    drops = set(drop_cols)
    df = df.drop(columns=drops)
    print('Coluna removida {}'.format(drops))
    return df

#--------------------------------------------------------------------------------------------------
#Implementando o pipeline de dados
#-------------------------------------------------------------------------------------------------- 
#Inicialização do caminho para os arquivos csv's
data_path_datasets = '/Users/Felipe/Documents/Python Scripts/Tese/Datasets'
data_path_predictions = '/Users/Felipe/Documents/Python Scripts/Tese/Predictions'
data_path_performances = '/Users/Felipe/Documents/Python Scripts/Tese/Performances'
data_path_times = '/Users/Felipe/Documents/Python Scripts/Tese/Times'

#Inicialização dos nomes dos datasets
datasets = ['banana.csv', 'haberman.csv','ionosphere.csv', #'hepatitis.csv']
            'SouthGermanCredit.csv', 'column_2C.csv']

#Definição da função para inserção no pipeline
drop_nan_pipeline = FunctionTransformer(drop_nan)
volumetria = FunctionTransformer(filtro_volumetria)
volatilidade = FunctionTransformer(filtro_volatilidade)
correlacao = FunctionTransformer(filtro_correlacao)

#Definição de um pipeline de dados
data_pipeline = Pipeline([('drop_nan', drop_nan_pipeline),
                          #('Volumetria', volumetria),
                          #('Volatilidade', volatilidade),
                          #('Correlação', correlacao),
                          ('Normalização', StandardScaler())])
                          
#Definição dos modelos avaliados
modelos_comparacao_sklearn = {'KNN': KNeighborsClassifier(),
                              'MLP': MLPClassifier(),
                              'Logística': LogisticRegression(),
                              'SVM': SVC()}

modelos = ['LSSVM', 'LSSVM_ADMM', 'CFGSMO_LSSVM', 'TCSMO_LSSVM',
           'P_LSSVM', 'IP_LSSVM', 'FSLM_LSSVM', 'FSLM_LSSVM_improved']

for dataset in datasets:

    #Carregando a base
    df = load_data(data_path_datasets, dataset)

    #Separando features de target
    X = df.iloc[:, range(df.shape[1]-1)]
    y = np.array(df.iloc[:, df.shape[1] -1])

    #Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    #Aplicando o pipeline de dados ao conjunto de treino
    X_train_processed = np.array(data_pipeline.fit_transform(X_train))

    #Aplicando o pipeline de dados ao conjunto de teste
    X_test_processed = np.array(data_pipeline.fit_transform(X_test))

    #----------------------------------------------------------------
    #----------------------------------------------------------------
    #Obtendo resultados para os modelos do sklearn
    #----------------------------------------------------------------
    #----------------------------------------------------------------
    #Ajuste dos modelos
    train_predict, test_predict = {}, {}
    time_processing = {}
    performance = {}
    for k in modelos_comparacao_sklearn:

        tic = process_time() #Inicialização do contador
        modelos_comparacao_sklearn[k].fit(X_train_processed, y_train)
        toc = process_time() #Finalização do contador

        #Armazenando as performance e os tempos de treinamento
        time_processing[k] = toc - tic
        train_predict[k] = modelos_comparacao_sklearn[k].predict(X_train_processed)
        test_predict[k] = modelos_comparacao_sklearn[k].predict(X_test_processed)

        #Obtendo as métricas  de performance
        performance[k] = metricas(y_test, test_predict[k])

    #---------------------------------------------------
    #Obtenção dos resultados para os modelos propostos
    #---------------------------------------------------
    #LSSVM
    tic = process_time()
    resultado_LSSVM = fit_LSSVM(X_train_processed, y_train, 0.5, "gaussiano")
    toc = process_time()
    
    alphas_LSSVM = resultado_LSSVM['mult_lagrange']
    b_LSSVM = resultado_LSSVM['b']
    train_predict['LSSVM'] = predict_class(alphas_LSSVM, b_LSSVM, "gaussiano", X_train_processed,
                                            y_train, X_train_processed)
    test_predict['LSSVM'] = predict_class(alphas_LSSVM, b_LSSVM, "gaussiano", X_train_processed,
                                            y_train, X_test_processed)
    performance['LSSVM'] = metricas(y_test, test_predict['LSSVM'])
    
    #Gerando os dataframes
    train_predict = pd.DataFrame(train_predict)
    test_predict = pd.DataFrame(test_predict)
    performance = pd.DataFrame(performance)
    time_processing = pd.Series(time_processing)

    #Exportando os resultados como excel
    train_predict.to_excel(f"train_predict_{dataset}.xlsx")
    test_predict.to_excel(f"test_predict_{dataset}.xlsx")
    performance.to_excel(f"performance_{dataset}.xlsx")
    time_processing.to_excel(f"time_processing_{dataset}.xlsx")











