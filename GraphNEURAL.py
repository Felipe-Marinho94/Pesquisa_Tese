#Apredendo a utilizar o pytorch geometrics
#Autor:Felipe Pinto Marinho
#Data:08/02/2023

#Carregando alguns pacotes relevantes

import os
import torch
from torch_geometric.datasets import Planetoid
import os.path as osp
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pytorch_forecasting
from JacobiConv.datasets import load_dataset
from JacobiConv.RealWorld import split
from JacobiConv.impl.PolyConv import PolyConvFrame
from JacobiConv.impl.PolyConv import JacobiConv, buildAdj
import torch.nn as nn
import numpy as np
import math
import tensorly as tl
from tensorly.tenalg.core_tenalg import inner
import pandas as pd

#Importando alguns pacotes
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random

#Configurando o backend
tl.set_backend('pytorch')
device = 'cpu'

#Carregando o dataset de imagens
dataset = load_dataset('photo')

#Printando o dataset
print(dataset.edge_index)
print(dataset.edge_attr)
print(dataset.num_classes)
print(dataset.num_nodes)
print(dataset.mask)
print(dataset.x)

#Criando um objeto da classe PolyConvFrame
Poly = PolyConvFrame(JacobiConv)
print(Poly.conv_fn)

#Construção de uma camada Jacobi
class JacobiLayer(nn.Module):
    
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_out)
        self.weights = nn.Parameter(weights)  
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # inicialização dos pesos e biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # Pesos init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        
        #Inicialização dos coeficientes do polinômio
        coef = nn.Parameter(torch.Tensor(Poly.alphas))

    def forward(self, x):
        w_times_x= torch.mm(x, self.weights)
        
        # Transformação Linear
        X = torch.add(w_times_x, self.bias)  # w times x + b
        
        #Filtragem com JacobiConv
        Z_hat = PolyConvFrame.forward(Poly, X, dataset.edge_index, dataset.edge_attr)
        
        return (Z_hat)


#Definindo a Camada de regressão Tensorial (TRL)
class TRL(nn.Module):
    def __init__(self, input_size, ranks, output_size, verbose=1, **kwargs):
        super(TRL, self).__init__(**kwargs)
        self.ranks = list(ranks)
        self.verbose = verbose

        if isinstance(output_size, int):
            self.input_size = [input_size]
        else:
            self.input_size = list(input_size)
            
        if isinstance(output_size, int):
            self.output_size = [output_size]
        else:
            self.output_size = list(output_size)
            
        self.n_outputs = int(np.prod(output_size[1:]))
        
        # Core of the regression tensor weights
        self.core = nn.Parameter(torch.Tensor(tl.zeros(self.ranks)), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(tl.zeros(1)), requires_grad=True)
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])
        
        # Add and register the factors
        self.factors = []
        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):
            self.factors.append(nn.Parameter(torch.Tensor(tl.zeros((in_size, rank))), requires_grad=True))
            self.register_parameter('factor_{}'.format(index), self.factors[index])
        
        # FIX THIS
        self.core.data.uniform_(-0.1, 0.1)
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        regression_weights = tl.tucker_to_tensor((self.core, self.factors))
        return inner(x, regression_weights, n_modes=tl.ndim(x)-1) + self.bias
    
    def penalty(self, order=2):
        penalty = tl.norm(self.core, order)
        for f in self.factors:
            penatly = penalty + tl.norm(f, order)
        return penalty


#Definindo a rede
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Jacobi1 = JacobiLayer(7650, 745)
        self.Jacobi2 = JacobiLayer(7650, 745)
        self.Jacobi3 = JacobiLayer(7650, 745)
        self.trl = TRL(input_size = (1,7650, 745, 3), ranks = (8, 3, 3, 8), output_size = (1, 8))
    
    def BuildTensor(self, X): #n: número de vértices, d: dimensão dos vetores de atributos
       
       #Tensor para armazenamento
       Tensor = torch.zeros([torch.Tensor.size(X)[0],
                             torch.Tensor.size(X)[1],
                            3])
       
       Tensor[:, :, 0] = self.Jacobi1.forward(X)
       Tensor[:, :, 1] = self.Jacobi2.forward(Tensor[:, :, 1])
       Tensor[:, :, 2] = self.Jacobi3.forward(Tensor[:, :, 2])

       return (Tensor)
   
    def forward(self, X):
        
        X = self.BuildTensor(X)
        X = self.trl.forward(torch.unsqueeze(X, dim=0))
        return F.log_softmax(X)

model = Net()
print(model)
Jacobi1 = JacobiLayer(7650, 745)
T = TRL(input_size = (1, 7650, 745, 3), ranks = (8, 3, 3, 8), output_size = (1, 8))
X = dataset.x
Tensor = model.BuildTensor(X)

te = T.forward(torch.unsqueeze(Tensor, dim = 0))
F.log_softmax(te)
Jacobi1.forward(X)
output =model(X)

#Definindo o otimizador
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion=nn.CrossEntropyLoss()

criterion(F.log_softmax(te), torch.rand(1,8))

#Definindo o treino e o teste
n_epoch = 5 # Number of epochs
regularizer = 0.001

model = model.to(device)

#Divisão treino-teste
Y = dataset.y
X = dataset.x
X = pd.DataFrame(X.numpy())
Y = pd.DataFrame(Y.numpy())
Dados = pd.concat([X, Y], axis = 1)

random.seed(2)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, shuffle = False)
X_train = torch.tensor(X_train.values.astype(np.float32))
y_train = torch.tensor(y_train.values.astype(np.float32))
train = torch.utils.data.TensorDataset(X_train, y_train)

X_test = torch.tensor(X_test.values.astype(np.float32))
y_test = torch.tensor(y_test.values.astype(np.float32))
test = torch.utils.data.TensorDataset(X_test, y_test)


#Obtenção de train_loader e test_loader utilizando
#torch_utils_DataLoader
batch_size = 16
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True) 
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True) 


def train(n_epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        
        data, target = sample[0], sample[1]
        data, target = data.to(device), target.to(device)
        
        # Important: do not forget to reset the gradients
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output,target) + regularizer*model.trl.penalty(2)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss = criterion(output,target)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('mean: {}'.format(test_loss))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))

train(2)

for epoch in range(1, n_epoch):
    train(epoch)
    test()

#Passando o tensor pela camada de Regressão Tensorial (TRL)
y = dataset.y
Y = y.type(torch.FloatTensor)
len(X_inicial)

#Criando um estimador
estimator = CPRegressor(weight_rank=2, tol=10e-7,
                        n_iter_max=50, reg_W=1, verbose=0)

#Ajustando o estimador aos dados
estimator.fit(Tensor, Y)

