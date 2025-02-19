import network
import mnist_loader as loader

# Load in MNIST Data for training

(train_data, valid_data, test_data) = loader.load_data_wrapper()

# print(train_data[0][0]) prints the vectorized image data
# print(train_data[0][1]) prints the vectorized label for the image

sizes = [784, 16, 16, 10]
nn = network.Network(sizes)

nn.SGD(train_data, 25, 10, 0.05)
print(str(nn.evaluate(test_data)) + " correct results out of " + str(len(test_data)))