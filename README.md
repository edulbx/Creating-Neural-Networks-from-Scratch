# Neural Network from Scratch

This project is an implementation of a basic neural network built entirely from scratch using Python. The code includes the construction of dense layers, activation functions, and a simple training process using backpropagation. It leverages popular libraries such as NumPy and Pandas for data manipulation and Scikit-learn for preprocessing. The network uses mean squared error as the loss function for training.

***there will be a code version and documentation in portuguese br and english***

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
  - [Dense Layer](#dense-layer)
  - [Activation Function](#activation-function)
  - [Network Model](#network-model)

## Installation

To run this project, ensure you have Python 3 installed. You can set up the environment and install the necessary packages using the following commands:

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-network-from-Scratch.git
cd neural-network-from-scratch

# Create a virtual environment
python -m venv env

# Activate the virtual environment
# On Windows
env\Scripts\activate

# On macOS and Linux
source env/bin/activate

# Install the required packages
pip install -r requirements.txt
The requirements.txt file should include the following packages and versions:

numpy==1.16.6
scipy==1.2.1
pandas==1.2.0
matplotlib==3.1.3
seaborn==0.11.1
scikit-learn==0.21.0
```

## Usage
To use the neural network, you need to create an instance of the Network class, add layers, and then train it using the provided data. Here is a quick start guide:
from neural_network import Network, Dense, ActLayer, relu, relu_prime, mse, mse_prime

# Create a network
```
net = Network(loss=mse, loss_prime=mse_prime)
```

# Add layers
```
net.add(Dense(feat_size=10, out_size=5))
net.add(ActLayer(act=relu, act_prime=relu_prime))
net.add(Dense(feat_size=5, out_size=1))
```
# Train the network
```
net.fit(X_train, y_train, epochs=1000, lr=0.01)
```
# Predict using the trained network
```
predictions = net.predict(X_test)
Make sure to replace X_train, y_train, and X_test with your actual data.
```
## Code Structure

# Dense Layer
The Dense class represents a fully connected layer in the neural network. It initializes weights and biases, performs forward passes, and updates weights during backpropagation.

# Activation Function
The ReLU (Rectified Linear Unit) activation function and its derivative are implemented to introduce non-linearity into the network.
```
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x
```
The ActLayer class is used to apply the activation function to the outputs of the dense layers.
class ActLayer:
```
    def __init__(self, act, act_prime):
        self.act = act
        self.act_prime = act_prime

    def forward(self, input_data):
        self.input = input_data
        self.output = self.act(self.input)
        return self.output

    def backward(self, output_der, lr):
        return self.act_prime(self.input) * output_der
```

# Network Model
The Network class is the core of the neural network, handling the addition of layers, forward passes, backpropagation, and training.

***for this documentation don't be too long, visit the code to see the network model***
