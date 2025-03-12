import network
import mnist_loader as loader

# Load in MNIST Data for training
(train_data, valid_data, test_data) = loader.load_data_wrapper()

# List of sizes of each layer of nodes
sizes = [784, 16, 16, 10]

# Create neural network
nn = network.Network(sizes)

# Train the neural network
learning_rate = 0.05
num_epochs = 25
mini_batch_size = 10
nn.SGD(train_data, num_epochs, mini_batch_size, learning_rate)

# Evaluate the model and print the correct results
print(str(nn.evaluate(test_data)) + " correct results out of " + str(len(test_data)))