import numpy as np

class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -709, 709)))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

class MLPClassifier:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', 
                 learning_rate=0.01, epochs=100, batch_size=32, optimizer='sgd', patience=5, min_delta=0.001, early_stopping=True):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.patience = patience
        self.min_delta = min_delta
        self.early_stopping = early_stopping

        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))

        self.set_activation(activation)

    def set_activation(self, activation):
        if activation == 'sigmoid':
            self.activation = ActivationFunction.sigmoid
            self.activation_derivative = ActivationFunction.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunction.tanh
            self.activation_derivative = ActivationFunction.tanh_derivative
        elif activation == 'relu':
            self.activation = ActivationFunction.relu
            self.activation_derivative = ActivationFunction.relu_derivative
        elif activation == 'linear':
            self.activation = ActivationFunction.linear
            self.activation_derivative = ActivationFunction.linear_derivative
        else:
            raise ValueError("Unsupported activation function")

    def forward_propagation(self, X):
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            a = self.activation(z) if i < len(self.weights) - 1 else ActivationFunction.softmax(z)
            self.layer_outputs.append(a)
        return self.layer_outputs[-1]

    def backward_propagation(self, X, y):
        m = X.shape[0]
        delta = self.layer_outputs[-1] - y
        gradients = []
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.layer_outputs[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients.append((dW, db))
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.layer_outputs[i])
        return list(reversed(gradients))

    def update_parameters(self, gradients):
        for i, (dW, db) in enumerate(gradients):
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def fit(self, X, y, X_val=None, y_val=None):
        self.losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                batch_X = X[i:i+self.batch_size]
                batch_y = y[i:i+self.batch_size]
                
                y_pred = self.forward_propagation(batch_X)
                gradients = self.backward_propagation(batch_X, batch_y)
                self.update_parameters(gradients)
            
            loss = self.compute_loss(X, y)
            self.losses.append(loss)
            
            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(X_val, y_val)
                self.val_losses.append(val_loss)
                
                if self.early_stopping:
                    if val_loss < best_val_loss - self.min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward_propagation(X), axis=1)

    def compute_loss(self, X, y):
        y_pred = self.forward_propagation(X)
        return -np.mean(y * np.log(y_pred + 1e-8))

    def gradient_checking(self, X, y, epsilon=1e-7):
        gradients = self.backward_propagation(X, y)
        for i, (dW, db) in enumerate(gradients):
            for j in range(dW.shape[0]):
                for k in range(dW.shape[1]):
                    self.weights[i][j, k] += epsilon
                    cost_plus = self.compute_loss(X, y)
                    self.weights[i][j, k] -= 2 * epsilon
                    cost_minus = self.compute_loss(X, y)
                    self.weights[i][j, k] += epsilon
                    
                    grad_approx = (cost_plus - cost_minus) / (2 * epsilon)
                    grad_backprop = dW[j, k]
                    
                    rel_error = abs(grad_backprop - grad_approx) / max(abs(grad_backprop), abs(grad_approx))
                    if rel_error > 1e-5:
                        print(f"Gradient Check Failed for W[{i}][{j},{k}]. Relative Error: {rel_error:.6f}")
                        return False
        
        print("Gradient Check Passed!")
        return True




class MultiLabelMLPClassifier:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', 
                 learning_rate=0.01, epochs=100, batch_size=32, optimizer='sgd', patience=5, min_delta=0.001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.patience = patience
        self.min_delta = min_delta

        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))

        self.set_activation(activation)

    def set_activation(self, activation):
        if activation == 'sigmoid':
            self.activation = ActivationFunction.sigmoid
            self.activation_derivative = ActivationFunction.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunction.tanh
            self.activation_derivative = ActivationFunction.tanh_derivative
        elif activation == 'relu':
            self.activation = ActivationFunction.relu
            self.activation_derivative = ActivationFunction.relu_derivative
        elif activation == 'linear':
            self.activation = ActivationFunction.linear
            self.activation_derivative = ActivationFunction.linear_derivative
        else:
            raise ValueError("Unsupported activation function")

    def forward_propagation(self, X):
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            a = self.activation(z) if i < len(self.weights) - 1 else ActivationFunction.sigmoid(z)
            self.layer_outputs.append(a)
        return self.layer_outputs[-1]

    def backward_propagation(self, X, y):
        m = X.shape[0]
        delta = self.layer_outputs[-1] - y
        gradients = []
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.layer_outputs[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients.append((dW, db))
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.layer_outputs[i])
        return list(reversed(gradients))

    def update_parameters(self, gradients):
        for i, (dW, db) in enumerate(gradients):
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def fit(self, X, y, X_val=None, y_val=None):
        self.losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                batch_X = X[i:i+self.batch_size]
                batch_y = y[i:i+self.batch_size]
                
                y_pred = self.forward_propagation(batch_X)
                gradients = self.backward_propagation(batch_X, batch_y)
                self.update_parameters(gradients)
            
            loss = self.compute_loss(X, y)
            self.losses.append(loss)
            
            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(X_val, y_val)
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return (self.forward_propagation(X) > 0.5).astype(int)

    def compute_loss(self, X, y):
        y_pred = self.forward_propagation(X)
        return -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))

    def gradient_checking(self, X, y, epsilon=1e-7):
        gradients = self.backward_propagation(X, y)
        for i, (dW, db) in enumerate(gradients):
            for j in range(dW.shape[0]):
                for k in range(dW.shape[1]):
                    self.weights[i][j, k] += epsilon
                    cost_plus = self.compute_loss(X, y)
                    self.weights[i][j, k] -= 2 * epsilon
                    cost_minus = self.compute_loss(X, y)
                    self.weights[i][j, k] += epsilon
                    
                    grad_approx = (cost_plus - cost_minus) / (2 * epsilon)
                    grad_backprop = dW[j, k]
                    
                    rel_error = abs(grad_backprop - grad_approx) / max(abs(grad_backprop), abs(grad_approx))
                    if rel_error > 1e-5:
                        print(f"Gradient Check Failed for W[{i}][{j},{k}]. Relative Error: {rel_error:.6f}")
                        return False
        
        print("Gradient Check Passed!")
        return True

import numpy as np

class MLPRegressor:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', 
                 learning_rate=0.01, epochs=100, batch_size=32, optimizer='sgd', patience=5, min_delta=0.001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.patience = patience
        self.min_delta = min_delta

        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))

        self.set_activation(activation)

    def set_activation(self, activation):
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_derivative = self.tanh_derivative
        elif activation == 'relu':
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation == 'linear':
            self.activation = self.linear
            self.activation_derivative = self.linear_derivative
        else:
            raise ValueError("Unsupported activation function")

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -709, 709)))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    def forward_propagation(self, X):
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            a = self.activation(z) if i < len(self.weights) - 1 else z
            self.layer_outputs.append(a)
        return self.layer_outputs[-1]

    def backward_propagation(self, X, y):
        m = X.shape[0]
        delta = self.layer_outputs[-1] - y.reshape(-1, 1)
        gradients = []
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.layer_outputs[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients.append((dW, db))
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.layer_outputs[i])
        return list(reversed(gradients))

    def update_parameters(self, gradients):
        for i, (dW, db) in enumerate(gradients):
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def fit(self, X, y, X_val=None, y_val=None):
        self.losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                batch_X = X[i:i+self.batch_size]
                batch_y = y[i:i+self.batch_size]
                
                y_pred = self.forward_propagation(batch_X)
                gradients = self.backward_propagation(batch_X, batch_y)
                self.update_parameters(gradients)
            
            loss = self.compute_loss(X, y)
            self.losses.append(loss)
            
            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(X_val, y_val)
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward_propagation(X)

    def compute_loss(self, X, y):
        y_pred = self.forward_propagation(X)
        return np.mean((y_pred - y.reshape(-1, 1))**2)

    def gradient_checking(self, X, y, epsilon=1e-7):
        gradients = self.backward_propagation(X, y)
        for i, (dW, db) in enumerate(gradients):
            for j in range(dW.shape[0]):
                for k in range(dW.shape[1]):
                    self.weights[i][j, k] += epsilon
                    cost_plus = self.compute_loss(X, y)
                    self.weights[i][j, k] -= 2 * epsilon
                    cost_minus = self.compute_loss(X, y)
                    self.weights[i][j, k] += epsilon
                    
                    grad_approx = (cost_plus - cost_minus) / (2 * epsilon)
                    grad_backprop = dW[j, k]
                    
                    rel_error = abs(grad_backprop - grad_approx) / max(abs(grad_backprop), abs(grad_approx))
                    if rel_error > 1e-5:
                        print(f"Gradient Check Failed for W[{i}][{j},{k}]. Relative Error: {rel_error:.6f}")
                        return False
        
        print("Gradient Check Passed!")
        return True

# Define the MLPClassifier_3 class for binary classification
class MLPClassifier_3:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', 
                 learning_rate=0.01, epochs=100, batch_size=32, optimizer='sgd', patience=5, min_delta=0.001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.patience = patience
        self.min_delta = min_delta

        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))

        self.set_activation(activation)

    def set_activation(self, activation):
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_derivative = self.tanh_derivative
        elif activation == 'relu':
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation == 'linear':
            self.activation = self.linear
            self.activation_derivative = self.linear_derivative
        else:
            raise ValueError("Unsupported activation function")

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -709, 709)))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    def forward_propagation(self, X):
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            a = self.activation(z) if i < len(self.weights) - 1 else self.sigmoid(z)
            self.layer_outputs.append(a)
        return self.layer_outputs[-1]

    def backward_propagation(self, X, y, loss_function='bce'):
        m = X.shape[0]
        if loss_function == 'bce':
            delta = self.layer_outputs[-1] - y.reshape(-1, 1)
        elif loss_function == 'mse':
            delta = (self.layer_outputs[-1] - y.reshape(-1, 1)) * self.sigmoid_derivative(self.layer_outputs[-1])
        gradients = []
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.layer_outputs[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients.append((dW, db))
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.layer_outputs[i])
        return list(reversed(gradients))

    def update_parameters(self, gradients):
        for i, (dW, db) in enumerate(gradients):
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def fit(self, X, y, X_val=None, y_val=None, loss_function='bce'):
        self.losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        y = y.to_numpy()  # Convert to numpy array
        if y_val is not None:
            y_val = y_val.to_numpy()  # Convert to numpy array

        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                batch_X = X[i:i+self.batch_size]
                batch_y = y[i:i+self.batch_size]
                
                y_pred = self.forward_propagation(batch_X)
                gradients = self.backward_propagation(batch_X, batch_y, loss_function)
                self.update_parameters(gradients)
            
            loss = self.compute_loss(X, y, loss_function)
            self.losses.append(loss)
            
            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(X_val, y_val, loss_function)
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward_propagation(X)

    def compute_loss(self, X, y, loss_function='bce'):
        y_pred = self.forward_propagation(X)
        if loss_function == 'bce':
            return -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        elif loss_function == 'mse':
            return np.mean((y_pred - y.reshape(-1, 1))**2)








import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MLP_combined:
    def __init__(self, input_size, hidden_sizes, output_size, task='regression', activation='relu', 
                 learning_rate=0.01, epochs=100, batch_size=32, optimizer='sgd', patience=5, min_delta=0.001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.task = task
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.patience = patience
        self.min_delta = min_delta

        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))

        self.set_activation(activation)

    def set_activation(self, activation):
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_derivative = self.tanh_derivative
        elif activation == 'relu':
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation == 'linear':
            self.activation = self.linear
            self.activation_derivative = self.linear_derivative
        else:
            raise ValueError("Unsupported activation function")

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -709, 709)))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    def forward_propagation(self, X):
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                a = self.activation(z)
            else:
                if self.task == 'classification':
                    a = self.sigmoid(z)
                else:
                    a = z
            self.layer_outputs.append(a)
        return self.layer_outputs[-1]

    def backward_propagation(self, X, y):
        m = X.shape[0]
        if self.task == 'classification':
            delta = self.layer_outputs[-1] - y.reshape(-1, 1)
        else:
            delta = self.layer_outputs[-1] - y.reshape(-1, 1)
        gradients = []
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.layer_outputs[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients.append((dW, db))
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.layer_outputs[i])
        return list(reversed(gradients))

    def update_parameters(self, gradients):
        for i, (dW, db) in enumerate(gradients):
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def fit(self, X, y, X_val=None, y_val=None):
        self.losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        y = y.to_numpy()  # Convert to numpy array
        if y_val is not None:
            y_val = y_val.to_numpy()  # Convert to numpy array

        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                batch_X = X[i:i+self.batch_size]
                batch_y = y[i:i+self.batch_size]
                
                y_pred = self.forward_propagation(batch_X)
                gradients = self.backward_propagation(batch_X, batch_y)
                self.update_parameters(gradients)
            
            loss = self.compute_loss(X, y)
            self.losses.append(loss)
            
            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(X_val, y_val)
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward_propagation(X)

    def compute_loss(self, X, y):
        y_pred = self.forward_propagation(X)
        if self.task == 'classification':
            return -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        else:
            return np.mean((y_pred - y.reshape(-1, 1))**2)

