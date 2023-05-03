from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import random
import math


# This is our class to define nodes in our neural network. First we have a constructor that takes in and saves all
# values we deemed important for a node to hold. Next we have methods to update each of these values individually
# when needed. Lastly, we have methods to calculate inputs, outputs, and error for each node in our network.
class Node:
    def __init__(self, weights, bias, net_input, output, error):
        self.weights = weights
        self.bias = bias
        self.net_input = net_input
        self.output = output
        self.error = error

    def updateWeights(self, weights):
        self.weights = weights

    def updateBias(self, bias):
        self.bias = bias

    def updateInput(self, net_input):
        self.net_input = net_input

    def updateOutput(self, output):
        self.output = output

    def calcNetInput(self, weights, train_row, bias):
        self.net_input = np.dot(weights, train_row) + bias

    def calcOutput(self, net_input):
        self.output = 1 / (1 + np.exp(-net_input))

    def calcOutputError(self, actual, output):
        self.error = output * (1 - output) * (actual - output)

    def calcHiddenError(self, output_error, output_weights, own_output):
        self.error = own_output * (1 - own_output) * (
                (output_error * output_weights[0]) + (output_error * output_weights[1]) + (
                output_error * output_weights[2]) + (output_error * output_weights[3]))


data = load_breast_cancer()
list(data.target_names)
['malignant', 'benign']
x = np.array(data.data)
y = np.array(data.target)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.7)
x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)
y_train = y_train.reshape((398, 1))
y_test = y_test.reshape((171, 1))


# This is our knn_distance method, and as the name suggests, it is responsible for
# calculating euclidean distance between
# two vectors. It works by iterating through each entry in both vectors and
# squares their difference. It then adds
# these values to difference to get a summation of all the vector's entries.
# Lastly, we return the square root of
# this summation.
def knn_distance(vector1, vector2):
    difference = 0
    for i, j in zip(vector1, vector2):
        difference = ((i - j) * (i - j)) + difference
    return math.sqrt(difference)


# This is our knn_predict method, and it is responsible for predicting the cancer cell
# classifications of our testing set. It
# works by first initializing some variables we will use later at their
# starting points. Next we iterate through our
# training set and find the closest training set vector to our input vector by
# calling our distance function. Once we
# find the closest one, we return the index of this training set vector.
def knn_predict(input_vector):
    current = 10000
    index = 0
    count = 0
    for i in x_train:
        if knn_distance(input_vector, i) < current:
            current = knn_distance(input_vector, i)
            index = count
        count += 1
    return index


# This is our logistic function. It is called in both our predict and gradient_descent methods as an activation function
def logistic_func(x1):
    return np.divide(1, (np.add(1, np.exp(-x1))))


# This is our gradient descent method. It works by first getting the shape of our x training set. It then runs a for
# loop for the number of epochs specified. Next it takes the dot product of our training set and the weights. This
# is then plugged into our logistic function. Next, we calculate our error and use this error to calculate our gradient
# Lastly, we update our weights and return the final form of them.
def gradient_descent(train_x, train_y, weight, lr, epochs):
    m, n = train_x.shape
    for i in range(epochs):
        predict1 = np.dot(train_x, weight)
        predicted = logistic_func(predict1)
        error = predicted - train_y
        grad_w = np.dot(train_x.T, error) / m
        weight = weight - lr * grad_w
    return weight


# This is our logistic regression predict method. It is responsible for making predictions on our testing set. It
# works by first taking the dot product of our x training set and our updated weights. It then plugs this number into
# our logistic function. Lastly, it replaces every prediction greater than or equal to 0.5 with a 1 and anything less
# with a 0 before returning our predictions.
def log_regress_predict(x_set, w):
    pred_1 = np.dot(x_set, w)
    pred_2 = logistic_func(pred_1)
    predictions = np.where(pred_2 >= 0.5, 1, 0)
    return predictions


# This is our initialize_Node method. It is responsible for initializing nodes weights and biases upon their creation in
# our neural network
def initialize_Node(size):
    for i in range(size):
        if i == 0:
            weights = np.array(random.randint(-1, 1))
        else:
            weights = np.append(weights, random.randint(-1, 1))
    bias = random.randint(-1, 1)
    return weights, bias


# This method is responsible for converting all our hidden layer's output into an array so it easier to use in other
# methods.
def createOutputArr():
    output_output = np.array(h1.output)
    output_output = np.append(output_output, h2.output)
    output_output = np.append(output_output, h3.output)
    output_output = np.append(output_output, h4.output)
    return output_output


# This method is responsible for training our neural network. It works by having an outer for loop that runs for the
# number of epochs specified and an inner for loop that iterates through each individual training sample. For each
# sample, it starts by calculating the hidden layer's net input followed by their outputs. Next, we calculate our
# output node's input and outputs. Next, calculate our errors for each node and back propagate to update each node's
# weight and biases accordingly.
def train_NN(epochs):
    m, n = x_train.shape
    for k in range(epochs):
        for i in range(m):
            current_sample = x_train[i]
            h1.calcNetInput(h1.weights, current_sample, h1.bias)
            h2.calcNetInput(h2.weights, current_sample, h2.bias)
            h3.calcNetInput(h3.weights, current_sample, h3.bias)
            h4.calcNetInput(h4.weights, current_sample, h4.bias)
            h1.calcOutput(h1.net_input)
            h2.calcOutput(h2.net_input)
            h3.calcOutput(h3.net_input)
            h4.calcOutput(h4.net_input)
            o_input = (h1.output * o1.weights[0]) + (h2.output * o1.weights[1]) + (h3.output * o1.weights[2]) + (
                    h4.output * o1.weights[3]) + o1.bias
            o1.updateInput(o_input)
            o1.calcOutput(o1.net_input)
            o1.calcOutputError(y_train[i], o1.output)
            h1.calcHiddenError(o1.error, o1.weights, h1.output)
            h2.calcHiddenError(o1.error, o1.weights, h2.output)
            h3.calcHiddenError(o1.error, o1.weights, h3.output)
            h4.calcHiddenError(o1.error, o1.weights, h4.output)
            h1_weight_change = calcWeightChange(current_sample, h1.error)
            h1_new_weights = h1.weights + h1_weight_change
            h1.updateWeights(h1_new_weights)
            h2_weight_change = calcWeightChange(current_sample, h2.error)
            h2_new_weights = h2.weights + h2_weight_change
            h2.updateWeights(h2_new_weights)
            h3_weight_change = calcWeightChange(current_sample, h3.error)
            h3_new_weights = h3.weights + h3_weight_change
            h3.updateWeights(h3_new_weights)
            h4_weight_change = calcWeightChange(current_sample, h4.error)
            h4_new_weights = h4.weights + h4_weight_change
            h4.updateWeights(h4_new_weights)
            o1_sample = createOutputArr()
            o1_weight_change = calcWeightChange(o1_sample, o1.error)
            o1_new_weights = o1.weights + o1_weight_change
            o1.updateWeights(o1_new_weights)
            h1_bias_change = calcBiasChange(h1.error)
            h1_new_bias = h1.bias + h1_bias_change
            h1.updateBias(h1_new_bias)
            h2_bias_change = calcBiasChange(h2.error)
            h2_new_bias = h2.bias + h2_bias_change
            h2.updateBias(h2_new_bias)
            h3_bias_change = calcBiasChange(h3.error)
            h3_new_bias = h3.bias + h3_bias_change
            h3.updateBias(h3_new_bias)
            h4_bias_change = calcBiasChange(h4.error)
            h4_new_bias = h4.bias + h4_bias_change
            h4.updateBias(h4_new_bias)
            o1_bias_change = calcBiasChange(o1.error)
            o1_new_bias = o1.bias + o1_bias_change
            o1.updateBias(o1_new_bias)


# This method is responsible for calculating the change in bias when we need to update a node's bias.
def calcBiasChange(own_error):
    return own_error * 0.05


# This method is responsible for calculating the change in weights when we need to update a node's weights.
def calcWeightChange(outputs, error):
    return 0.05 * error * outputs


# This method is responsible for making predictions from neural network for our testing set. It works very similarly
# to our training method except it doesn't do any back propagation to update any weights or biases. When it
# calculates our output of our output layer it rounds its prediction accordingly to one of our binary options (0 or
# 1). Lastly, we return these predictions.
def nn_predict():
    m, n = x_test.shape
    predictions = []
    for i in range(m):
        current_sample = x_test[i]
        h1.calcNetInput(h1.weights, current_sample, h1.bias)
        h2.calcNetInput(h2.weights, current_sample, h2.bias)
        h3.calcNetInput(h3.weights, current_sample, h3.bias)
        h4.calcNetInput(h4.weights, current_sample, h4.bias)
        h1.calcOutput(h1.net_input)
        h2.calcOutput(h2.net_input)
        h3.calcOutput(h3.net_input)
        h4.calcOutput(h4.net_input)
        o_input = (h1.output * o1.weights[0]) + (h2.output * o1.weights[1]) + (h3.output * o1.weights[2]) + (
                h4.output * o1.weights[3]) + o1.bias
        o1.updateInput(o_input)
        o1.calcOutput(o1.net_input)
        if o1.output >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# This is our final_predict method, and it is responsible for finding the most common prediction on our testing set,
# when tested against all three models. It outputs an array containing the final predictions.
def final_predict(knn, nn, logr):
    final_predictions = []
    one_check = np.array([1])
    zero_check = np.array([0])
    for i in range(len(knn)):
        all_result = [knn[i], nn[i], logr[i]]
        mode, count = stats.mode(all_result, keepdims=True)
        if mode == one_check:
            final_predictions.append(1)
        elif mode == zero_check:
            final_predictions.append(0)

    return final_predictions


# This is our formatting method, and it is responsible for getting the output of our three models all in a similar
# data structure so comparison between results is easier.
def formatting(knn, logr):
    reformat_knn = []
    reformat_logr = []
    one_check = np.array([1])
    zero_check = np.array([0])
    for i in range(len(knn)):
        if knn[i] == one_check:
            reformat_knn.append(1)
        elif knn[i] == zero_check:
            reformat_knn.append(0)
    for j in range(len(logr)):
        if logr[j] == one_check:
            reformat_logr.append(1)
        elif logr[j] == zero_check:
            reformat_logr.append(0)
    return reformat_knn, reformat_logr


w1, b1 = initialize_Node(30)
h1 = Node(w1, b1, 0, 0, 0)
w2, b2 = initialize_Node(30)
h2 = Node(w2, b2, 0, 0, 0)
w3, b3 = initialize_Node(30)
h3 = Node(w3, b3, 0, 0, 0)
w4, b4 = initialize_Node(30)
h4 = Node(w4, b4, 0, 0, 0)
w5, b5 = initialize_Node(4)
o1 = Node(w5, b5, 0, 0, 0)


# This is our main method. It trains all three models and stores their predictions on the testing set. It also outputs
# each model's accuracy individually before finding the final predictions and outputting those predictions and accuracy.
def main():
    print('running KNN')
    knn_predictions = []
    for h in x_test:
        output = knn_predict(h)
        knn_predictions.append(y_train[output])
    knn_score = accuracy_score(knn_predictions, y_test)
    print("KNN Accuracy Score: ", knn_score)
    print('running logistic regression (may take some time)')
    x_train_rows, x_train_columns = x_train.shape
    x_test_rows, x_test_columns = x_test.shape
    lr_x_train = x_train
    lr_x_test = x_test
    train_intercepts = np.ones(x_train_rows)
    train_intercepts = train_intercepts.reshape((x_train_rows, 1))
    test_intercepts = np.ones(x_test_rows)
    test_intercepts = test_intercepts.reshape((x_test_rows, 1))
    updated_x_train = np.hstack((lr_x_train, train_intercepts))
    updated_x_test = np.hstack((lr_x_test, test_intercepts))
    rows, columns = updated_x_test.shape
    weights = np.array(np.zeros(columns))
    weights = weights.reshape((31, 1))
    weights = gradient_descent(updated_x_train, y_train, weights, 0.05, 100000)
    log_predictions = log_regress_predict(updated_x_test, weights)
    log_score = accuracy_score(log_predictions, y_test)
    print("Logistic Regression Accuracy Score: ", log_score)
    print('running neural network (may take some time)')
    train_NN(1000)
    nn_predictions = nn_predict()
    nn_score = accuracy_score(nn_predictions, y_test)
    print("Neural Network Accuracy Score: ", nn_score)
    log_predictions_list = log_predictions.tolist()
    formatted_knn, formatted_log = formatting(knn_predictions, log_predictions_list)
    final_predictions = final_predict(formatted_knn, nn_predictions, formatted_log)
    print('knn predictions', formatted_knn)
    print('nnn predictions', nn_predictions)
    print('log predictions', formatted_log)
    print('fin predictions', final_predictions)
    overall_score = accuracy_score(final_predictions, y_test)
    print("Ensemble Model Accuracy Score: ", overall_score)


if __name__ == '__main__':
    main()
