import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# -------------------------------------------------- FUNCTIONS ------------------------------------------------------ #

# -------------- CREATE RANDOM WEIGHTS ------------- #

def init_params():
    input_weight = np.random.rand(10, 784) - 0.5  # Weight of the input neurons.
    hidden_weight = np.random.rand(10, 10) - 0.5  # Weight of the hidden layer neurons.
    input_bias = np.random.rand(10, 1) - 0.5  # Bias of the input neurons.
    hidden_bias = np.random.rand(10, 1) - 0.5  # Bias of the hidden layer neurons.
    return input_weight, input_bias, hidden_weight, hidden_bias
    
    # input_weight is a 10 row, 784 column of random numbers between -0.5 and 0.5
    # [[-0.49099459  0.31193223 -0.46602156 ... -0.33906885 -0.45778696 0.29457142]
    # ...
    # [-0.46644467 -0.32010455 -0.4060165  ...  0.18024616 -0.4851882 0.16641178]]
    
    # input_bias is a 10 row, 1 column of random numbers. So for each row of weights have one bias.
    # [[ 0.34142552]
    # ...
    #  [ 0.19385157]]


def ReLU(Z):
    return np.maximum(Z, 0)  # This will return Z if greater than 0. Return 0 if lower than 0.
    # ReLU (Rectified Linear Unit) activation function.
    # f(x) = x (if x > 0)
    # f(x) = 0 (if x < 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    # Softmax or Normalized Exponential Function
    # Basically what this does is: Calculate all the element's exponential function with Z and divide it to the sum of
    # all element's exponential function of Z.


def forward_prop(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1  # dot is a "matrix multiply operation". Basically, all weights interact with pixel values.
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
    # Basically the mathematical function we need works like this:
    # We need to multiply the weights with the data and add bias on it. This represents the reaction of neurons.
    # w1.dot(X) + b1  > This is basically neurons' values after you multiply their weights with the data.
    # Then you send it to ReLU() function. This will get rid of negative numbers.
    # Neurons will have their values bigger than 0, and we will pass this values into the hidden layer neurons
    # by applying dot operation again.
    # Z2 = w2.dot(A1) + b2  > Hidden layer's neurons' values.
    # And then we apply the last activation function on it to normalise its values.
    # A2 = softmax(Z2)  > This is the final output values we want.


def deriv_ReLU(Z):
    return Z > 0  # This part is about the slope of the function. This will return 1 (True) or 0 (False).


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # Y.size is "m", how many examples there are.
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
    # one_hot is an encoding that uses indexes. If you one_hot encode "[1, 2, 3]", it would look like this:
    # [
    #   [0, 1, 0, 0],
    #   [0, 0, 1, 0],
    #   [0, 0, 0, 1]
    # ]
    # np.zeros((Y.size, Y.max() + 1))
    # So we basically created a matrix of zeros with rows (example times) and columns (10). Around 60.000 rows
    # and 10 columns of zeros.
    # np.arange(Y.size) is an array starting from 0 to 60.000≈. Then we set the specific row and column of
    # these zeros to 1.
    # one_hot_Y[np.arange(Y.size), Y] = 1  > This will one_hot encode all numbers in answers.

    # For example: zeros[np.arange(5), [3, 4, 2, 1, 0]] = 1  > Set each column of zeros to 1 from row 0 to row 10.
    # 00010
    # 00001
    # 00100
    # 01000
    # 10000
    
    # Then we transpose it, so it's a 10 row and around 60.000 columns of zero except the ones we set to 1.


def back_prop(Z1, A1, Z2, A2, w1, w2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dw2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    
    dZ1 = w2.T.dot(dZ2) * deriv_ReLU(Z1)
    dw1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dw1, db1, dw2, db2
    # This section is just mathematical. In short, this will be used to slightly change the weights of Neurons in the
    # correct direction.


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1  # if alpha is big, weights will change more.
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2  # This function will rearrange the weights of connections between Neurons.


def get_predictions(A2):
    return np.argmax(A2, 0)  # Returns the neuron that has the biggest value. The answer of our Neural Network.


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def start(X, Y, iterations, alpha):
    try:  # Load existing Neural Network, if it doesn't exit create one.
        w1 = np.load("TrainedWeights/Input Weight.npy")
        w2 = np.load("TrainedWeights/Hidden Weight.npy")
        b1 = np.load("TrainedWeights/Input Bias.npy")
        b2 = np.load("TrainedWeights/Hidden Bias.npy")
    except FileNotFoundError:
        w1, b1, w2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b1, X)
        dw1, db1, dw2, db2 = back_prop(Z1, A1, Z2, A2, w1, w2, X, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 50 == 0:
            print(f"Iteration: {i}")
            predictions = get_predictions(A2)
            print(f"Accuracy: {round(get_accuracy(predictions, Y) * 100)}%")
    return w1, b1, w2, b2


# ------------------------------------------- READ DATA AND CREATE ARRAY -------------------------------------------- #

data = pd.read_csv("Data/mnist_train.csv")  # Read training data.
data = np.array(data)  # To be able to do calculations on data, we turn it into a Numpy array.
np.random.shuffle(data)  # Shuffling the data prevents the Neural Network to over fit.
m, n = data.shape  # m is the rows, n is the columns (pixel values in this situation)

# Right now the data looks like this:
# [[9 0 0 ... 0 0 0]  > 785 columns for each pixel of the drawings, first column is the label.
#  [8 0 0 ... 0 0 0]  > Rows as much as examples (Around 60.000)
#  ...
#  [1 0 0 ... 0 0 0]
#  [2 0 0 ... 0 0 0]]

# ------------------------------------------- CREATE TWO NEW DATA SETS ---------------------------------------------- #

# -------------- DATA FOR TESTING ------------- #

data_dev = data[0:1000].T  # We create a new data from the first 1000 data. And then "T" transpose it, means rows
# become columns and columns become rows.

# [[9 8 5 ... 4 1 2]  > Now the labels becomes the top row.
#  [0 0 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]]

answers_test = data_dev[0]  # First row (9 8 5 ... 4 1 2)
pixels_test = data_dev[1:n]  # Rest of the rows
pixels_test = pixels_test / 255.  # Since calculating bigger numbers would be more complex and difficult we divide
# all elements to 255. So instead of having color values between 0 and 255, we have values between 0 and 1.

# ------------ DATA FOR TRAINING ------------- #

data_train = data[1000:m].T  # Again we create another data from the rest, 1000th row to the end.
answers = data_train[0]  # This is the first row
pixels = data_train[1:n]  # Rest of the rows
pixels = pixels / 255.  # To make calculation easier.
_, m_train = pixels.shape


# ------------------------------------------- RUN THE NEURAL NETWORK ------------------------------------------------ #

ITERATION = 2
LEARNING_CURVE = 0.1

# Send these data to start function and assign the returned parameters to save the Neural Connections.
weight_input, bias_input, weight_hidden, bias_hidden = start(pixels, answers, ITERATION, LEARNING_CURVE)

# -------- Save the Network --------- #
np.save("TrainedWeights/Input Weight.npy", weight_input)
np.save("TrainedWeights/Hidden Weight.npy", weight_hidden)
np.save("TrainedWeights/Input Bias.npy", bias_input)
np.save("TrainedWeights/Hidden Bias.npy", bias_hidden)


# -------------------------------------------- DISPLAY SOME EXAMPLES ------------------------------------------------ #

def make_predictions(X, w1, b1, w2, b2):
    _, _, _, A2 = forward_prop(w1, b1, w2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, w1, b1, w2, b2):
    current_image = pixels[:, index, None]
    prediction = make_predictions(pixels[:, index, None], w1, b1, w2, b2)
    label = answers[index]

    current_image = current_image.reshape((28, 28)) * 255  # To show the data as an image in Matplotlib

    plt.gray()
    plt.imshow(current_image, aspect='auto', interpolation='nearest')
    plt.title(f"Label: {label} | Prediction: {prediction}")
    plt.axis('off')
    manager = plt.get_current_fig_manager()
    manager.window.setGeometry(480, 100, 650, 650)
    plt.show()


# Some random images to show how successful our Neural Network is.
test_prediction(1, weight_input, bias_input, weight_hidden, bias_hidden)
test_prediction(2, weight_input, bias_input, weight_hidden, bias_hidden)
test_prediction(3, weight_input, bias_input, weight_hidden, bias_hidden)
test_prediction(4, weight_input, bias_input, weight_hidden, bias_hidden)
test_prediction(5, weight_input, bias_input, weight_hidden, bias_hidden)
test_prediction(6, weight_input, bias_input, weight_hidden, bias_hidden)
test_prediction(7, weight_input, bias_input, weight_hidden, bias_hidden)
test_prediction(8, weight_input, bias_input, weight_hidden, bias_hidden)
test_prediction(9, weight_input, bias_input, weight_hidden, bias_hidden)
test_prediction(10, weight_input, bias_input, weight_hidden, bias_hidden)

# To test with a different data set.
# dev_predictions = make_predictions(pixels_test, weight_input, bias_input, weight_hidden, bias_hidden)
# get_accuracy(dev_predictions, answers_test)
# print(f"Accuracy: {round(get_accuracy(dev_predictions, answers_test) * 100)}%")
