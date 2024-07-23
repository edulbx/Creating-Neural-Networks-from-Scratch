# Importing the lib
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# numpy     : 1.16.6
# sklearn   : 0.21.0
# pandas    : 1.2.0
# matplotlib: 3.1.3
# scipy     : 1.2.1
# seaborn   : 0.11.1

# Classe para a camada densa
class Dense:
    
    # Método construtor
    def __init__(self, feat_size, out_size):
        self.feat_size = feat_size
        self.out_size = out_size
        self.weights = (np.random.normal(0, 1, feat_size * out_size) * np.sqrt(2 / feat_size)).reshape(feat_size, out_size)
        self.bias = np.random.rand(1, out_size) - 0.5

    # Método da passada linear para frente
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return(self.output)

    # Método da passada de volta (backpropagation)
    def backward(self, output_der, lr): 
        input_der = np.dot(output_der, self.weights.T)
        weight_der = np.dot(self.input.T.reshape(-1, 1), output_der)
        self.weights -= lr * weight_der
        self.bias -= lr * output_der
        return(input_der)
    

# Função de ativação
def relu(x):  
    return(np.maximum(0, x))

# Derivada da função de ativação
def relu_prime(x):  
    x[x > 0] = 1
    x[x <= 0] = 0  
    return x

# Classe da camada de ativação
class ActLayer:
    
    # Método construtor
    def __init__(self, act, act_prime):
        self.act = act
        self.act_prime = act_prime

    # Recebe a entrada (input) e retorna a saída da função de ativação
    def forward(self, input_data): 
        self.input = input_data 
        self.output = self.act(self.input)
        return(self.output)

    # Observe que não estamos atualizando nenhum parâmetro aqui
    # Usamos a taxa de aprendizagem como parâmetro porque definiremos o método de ajuste de uma forma 
    # que todas as camadas o exigirão.
    def backward(self, output_der, lr):
        return(self.act_prime(self.input) * output_der)

# Usaremos a Mean-Squared-Error como função de perda
def mse(y_true, y_pred):
    return(np.mean((y_pred - y_true)**2))

# Derivada da função de perda
def mse_prime(y_true, y_pred):
    return(2*(y_pred - y_true) / y_true.size)

# Modelo
class Network:
    
    # Método construtor
    # Inicializa com a função de perda e sua derivada
    def __init__(self, loss, loss_prime): 
        #problema de otimização matematica, 
        #quer sempre encontar o menor erro possível, por isso inicia com a função de perda.  
        self.layers = []  
        self.loss = loss
        self.loss_prime = loss_prime

    # Método para adicionar camadas ao grafo computacional
    def add(self, layer):
        self.layers.append(layer) #nenhuma operação matematica avançada, ou dificil.

    # Implementando apenas forward-pass para predição
    def predict(self, input_data): #sempre que ver um função verifique entrada e saida
        #aqui ela recebe dado d entrada e retorna um
        
        # Lista para o resultado
        result = [] 

        for a in range(len(input_data)):
            
            # Camada de saída
            layer_output = input_data[a]
            
            # Loop pelas camadas
            for layer in self.layers:
                
                # Movendo vetores de camada para camada
                layer_output = layer.forward(layer_output) #vai avannçadno em cada uma das
                #camadas, por isso chama o fwd
                
            result.append(layer_output)

        return(result)

    # Método de treinamento
    def fit(self, X_train, y_train, epochs, lr): #metodo feito para treinamento
        #epochs são número de epocas e lr taxa de aprendizado, cada época é uma
        #passagem de treinamento

        # Número de iterações
        for a in range(epochs):  
            
            # Inicializa a variável de cálculo do erro
            err = 0

            # Temos 1 passagem para a frente e para trás para cada ponto de dados 
            # Esse algoritmo de aprendizagem usa a Descida Estocástica do Gradiente
            for j in range(len(X_train)):
                
                # Camada de saída
                layer_output = X_train[j]
                
                # Loop pelas camadas
                for layer in self.layers:
                    layer_output = layer.forward(layer_output)

                # Vamos guardar o erro e mostrar durante o treinamento
                err += self.loss(y_train[j], layer_output) #calcula com a previsão do modelo 
                #mais o y_train q é o valor real.Esse é o foward, precisa fazer o back

                # Observe que fazemos o loop nas camadas em ordem reversa.
                # Inicialmente calculamos a derivada da perda com relação à previsão.
                # Em seguida, a camada de saída irá calcular a derivada em relação à sua entrada
                # e irá passar esta derivada de entrada para a camada anterior que corresponde à sua derivada de saída
                # e essa camada repetirá o mesmo processo, passando sua derivada de entrada para a camada anterior.

                # dL/dY_hat
                gradient = self.loss_prime(y_train[j], layer_output)
                #na volta tá usando a loss_prime derivada da funação de perda;
                
                # Este loop é a razão de termos dado lr à camada de ativação como argumento
                for layer in reversed(self.layers): #loopreverso reversed
                    
                    # Definindo gradiente para dY / dh_ {i + 1} da camada atual
                    gradient = layer.backward(gradient, lr)

            err /= len(X_train)
            
            print('Epoch %d/%d   Erro = %f' % (a + 1, epochs, err))
