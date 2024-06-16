from NeuralNetworks import NeuralNetwork
import pandas as pd
import numpy as np

def load_data(file_path):
    # Read the CSV file
    data_df = pd.read_csv(file_path).dropna()
    
    # Extract the labels
    labels = data_df.iloc[:, -1].values
    
    # Extract the inputs
    inputs = data_df.iloc[:, :-1].values
    
    # Normalize the inputs
    #inputs = inputs / 255.0
    
    # One-hot encode the labels
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels] = 1
    
    return inputs, one_hot_labels

# Load the training data
train_inputs, train_labels = load_data("MNIST-train.csv")

# Initialize the neural network with the given architecture
network = NeuralNetwork([784, 30, 20, 10])

# Train the neural network
network.fit(train_inputs, train_labels, epochs=100, mini_batch_size=20, eta=4)

predicted = network.predict_array(train_inputs[:100])

predicted_and_label = [(predicted[i],np.argmax(train_labels[i])) for i in range(100)]

print(predicted_and_label)

print(network.score(train_inputs[:100],train_labels[:100]))
