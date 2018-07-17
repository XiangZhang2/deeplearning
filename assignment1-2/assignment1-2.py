import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


def sigmoid(z):
    
    s = 1 / (1 + np.exp(-z))
    return s

# print sigmoid(np.array([0, 2]))


def forward(w, b, X, Y):
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m
    cost = np.squeeze(cost)
    return A, cost

# w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
# A, cost = forward(w,b, X, Y)
# print A
# print cost


def backward(X, A, Y):
    
    m = X.shape[1]
    dw = (np.dot(X, (A - Y).T)) / m
    db = (np.sum(A - Y)) / m
    return dw, db

# dw, db = backward(X, A, Y)
# print dw, db


def optimize(w, b, X, Y, num_iterations, learning_rate):
    
    costs = []
    
    for i in range(num_iterations):
        A, cost = forward(w, b, X, Y)
        # print cost
        dw, db = backward(X, A, Y)
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
        
        if i % 100 == 0:
            costs.append(cost)

    return w, b, costs

# w, b, costs = optimize(w, b, X, Y, num_iterations= 101, learning_rate = 0.009)
# print w, b, costs


def predict(w, b, X):
    
    A = sigmoid(np.dot(w.T, X) + b)
    Y_prediction = np.zeros((1, X.shape[1]))
    
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction

# print ("predictions = " + str(predict(w, b, X)))


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5):
    
    w, b = np.zeros((train_set_x.shape[0], 1)), 0
    w, b, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "w": w,
         "b": b, 
         "learning_rate": learning_rate,
         "num_iterations":num_iterations}
    return d


if __name__ == '__main__':
    
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    train_set_x_flatten = train_set_x_orig.reshape((m_train, num_px * num_px * 3)).T
    test_set_x_flatten = test_set_x_orig.reshape((m_test, num_px * num_px * 3)).T

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    d = model(train_set_x, train_set_y, test_set_x,test_set_y, num_iterations = 2000, learning_rate = 0.005)
    costs = np.squeeze(d["costs"])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()

    ###########  test different lr  ##############
    # learning_rates = [0.01, 0.005, 0.001, 0.0001]
    # models = {}
    # for i in learning_rates:
    #     print ("learning rate is: " + str(i))
    #     models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i)
    #     print ('\n' + "-------------------------------------------------------" + '\n')

    # for i in learning_rates:
    #     plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

    # plt.ylabel('cost')
    # plt.xlabel('iterations')

    # legend = plt.legend(loc='upper center', shadow=True)
    # frame = legend.get_frame()
    # frame.set_facecolor('0.90')
    # plt.show()


    ######  Test your own image  ############
    # my_image = "zhu.jpg"
    # filename = "images/" + my_image
    # image = np.array(ndimage.imread(filename, flatten=False))
    # my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    # my_predicted_image = predict(d["w"], d["b"], my_image)

    # plt.imshow(image)
    # plt.show()
    # print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
















