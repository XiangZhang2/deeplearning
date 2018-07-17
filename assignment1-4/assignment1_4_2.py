import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

num_px = train_x_orig.shape[1]


def two_layer_model(X, Y, layers_dims, num_iterations, learning_rate, print_cost, show_plt):
    grads = {}
    costs = []
    #parameters = initialize_parameters_deep(layers_dims)
    parameters = initialize_parameters(12288, 7, 1)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, activation = "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation = "sigmoid")

        cost = compute_cost(A2, Y)

        #dA2 = -(Y / A2) - ((1 - Y) / (1 - A2))
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation = "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation = "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if print_cost and i % 100 == 0:
            print ("i %d: cost %.10f:" % (i, np.squeeze(cost)))
            costs.append(cost)

    if show_plt:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


# def predict(X, Y, parameters):
#     m = X.shape[1]
#     A2, caches = L_model_forward(X, parameters)
#     predictions = np.zeros((1, m))
    
#     for i in range(m):
#         if A2[0, i] > 0.5:
#             predictions[0, i] = 1
#         else:
#             predictions[0, i] = 0
    
#     print ("Accuracy: "  + str(np.sum((predictions == Y)/float(m))))

#     return predictions


def two_layers_model_test():
    parameters = two_layer_model(train_x, train_y, layers_dims=(12288, 7, 1), num_iterations=2500, learning_rate=0.0075, print_cost=True, show_plt=False)
    predictions_train = predict(train_x, train_y, parameters)
    predictions_test = predict(test_x, test_y, parameters)


def L_layer_model(X, Y, layers_dims, num_iterations, learning_rate, print_cost):
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def L_layers_model_test():
    layers_dims = [12288, 20, 7, 5, 1]
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, learning_rate=0.0075, print_cost = True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)


def test_own_image():
    layers_dims = [12288, 20, 7, 5, 1]
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, learning_rate=0.0075, print_cost = True)
    my_image = "zhu.jpg"
    my_label_y = [1]
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
    my_predicted_image = predict(my_image, my_label_y, parameters)
    plt.imshow(image)
    plt.show()
    print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")





























