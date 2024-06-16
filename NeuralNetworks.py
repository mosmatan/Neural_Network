import random
import numpy as np
from numpy import array

class NeuralNetwork:
    def __init__(self,layers_sizes ,activation_func = 'tanh') -> None:
        
        self.num_of_layers = len(layers_sizes)
        
        # Initialize weights randomly with a normal distribution
        self.weights = [np.random.randn(x,y) for x,y in zip(layers_sizes[:-1],layers_sizes[1:])]
        
        # Initialize biases randomly with a normal distribution
        self.biases = [np.random.randn(size,1) for size in layers_sizes[1:]]
        
        if activation_func == 'tanh':
            self.activation_func = self.tanh
            self.activation_func_derivative = self.tanh_derivative
            
        elif activation_func == 'sigmoid':
            self.activation_func = self.sigmoid
            self.activation_func_derivative = self.sigmoid_derivative
    
    def tanh(self, z):
        e_plus, e_minus = np.exp(z), np.exp(-z)
        return (e_plus - e_minus) / (e_plus + e_minus)
    
    def tanh_derivative(self, z):
        return 1 - self.tanh(z)**2
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def feedforward(self, s):
        #Perform the feedforward operation.
        s=s.reshape(1,s.shape[0])
        for b, w in zip(self.biases, self.weights):
            s = self.activation_func(s @ w + b.T)
        return s
    
    def cost_derivative(self, output_activations, y):
        return output_activations - y
    
    def backpropagate(self, x, y):
         
         current_x = x.reshape(x.shape[0],1)
         x_by_layers = [current_x]
         z_by_layers = []
         
         for w,b in zip(self.weights,self.biases):
             current_z = (current_x.T @ w).T + b
             z_by_layers.append(current_z)
             current_x = self.activation_func(current_z)
             x_by_layers.append(current_x)
             
        # Initialize gradients
         gradients_b = [np.zeros(b.shape) for b in self.biases]
         gradients_w = [np.zeros(w.shape) for w in self.weights]
         
         y= y.reshape(y.shape[0],1)
         delta = self.cost_derivative(x_by_layers[-1], y) * self.activation_func_derivative(z_by_layers[-1])
         gradients_b[-1] = delta
         gradients_w[-1] = x_by_layers[-2] @ delta.T  
         
         for l in range(2, self.num_of_layers):
            z = z_by_layers[-l]
            sd = self.activation_func_derivative(z)
            delta = self.weights[-l + 1] @ delta * sd
            gradients_b[-l] = delta
            gradients_w[-l] = x_by_layers[-l-1] @ delta.T
            
         return gradients_w, gradients_b
     
    def update_mini_batch(self, mini_batch, eta):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_grad_w, delta_grad_b = self.backpropagate(x, y)
            
            grad_w = [weight_grad + delta_weight_grad for weight_grad, delta_weight_grad in zip(grad_w, delta_grad_w)]
            grad_b = [bias_grad + delta_bias_grad for bias_grad, delta_bias_grad in zip(grad_b, delta_grad_b)]
           
        self.weights = [w - (eta / len(mini_batch)) * weight_grad for w, weight_grad in zip(self.weights, grad_w)]
        self.biases = [b - (eta / len(mini_batch)) * bias_grad for b, bias_grad in zip(self.biases, grad_b)]
        
    def create_mini_batches(self, inputs, labels, mini_batch_size):
        data = list(zip(inputs, labels))
        random.shuffle(data)
        mini_batches = [data[k:k + mini_batch_size] for k in range(mini_batch_size)]
        return mini_batches
    
    def fit(self, X, y, epochs, mini_batch_size, eta):
        max_value = np.max(X)
        X = X / max_value
        
        for epoch in range(epochs):
            mini_batches = self.create_mini_batches(X, y, mini_batch_size)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print(f"Epoch {epoch + 1} complete")
            
    def predict(self, X):
        return np.argmax(self.feedforward(X))
    
    def predict_array(self, X):
        return [self.predict(x) for x in X]
    
    def score(self, X, y):
        """Compute the accuracy of the neural network."""
        predictions = [self.predict(x) for x in X]
        true_labels = [np.argmax(label) for label in y]
        accuracy = sum(int(pred == true) for pred, true in zip(predictions, true_labels)) / len(y)
        return accuracy