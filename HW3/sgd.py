#################################
# Your name: Lihu Zur
#################################


import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import scipy.special
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    n = len(data[0])
    w = np.zeros(n)

    for t in range(1, T + 1):
        eta_t = eta_0 / t
        i = np.random.randint(len(data))
        x_i = data[i]
        y_i = labels[i]

        if y_i * (x_i @ np.transpose(w)) < 1:
            w = (1 - eta_t) * w + eta_t * C * y_i * x_i
        else:
            w = (1 - eta_t) * w

    return w


def SGD_log(data, labels, eta_0, T, draw_norms):
    """
    Implements SGD for log loss.
    the gradient of log(1+e^(-ywx)) w.r. to w is ( (-y*e^(-yxw)) / (1+e^(-yxw)) ) * x
    """
    n = len(data[0])
    w = np.zeros(n)
    norms = []

    for t in range(1, T+1):
        i = np.random.randint(len(data))
        x_i = data[i]
        y_i = labels[i]
        eta_t = eta_0 / t  # Liad wrote in the forum to add this also in log loss
        w = w - eta_t * calc_grad_llog(w, x_i, y_i)  # Gradient descent step with the gradient of llog
        norms.append(np.linalg.norm(w))

    # Draw the norms graph for 2c
    if draw_norms:
        plt.plot([i for i in range(T)], norms)
        plt.xlabel("iteration")
        plt.ylabel("norm of w")
        plt.show()

    return w


#################################

# Place for additional code

#################################


def q1_a(train_data, train_labels, validation_data, validation_labels, etas):
    avg_acc = []

    for eta in etas:
        eta_acc = []
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, 1, eta, 1000)
            calc = calc_successes(w, validation_data, validation_labels)  # Counting successful classifications
            eta_acc.append(calc)

        avg_acc.append(np.mean(eta_acc))

    plt.plot(etas, avg_acc)
    plt.xlabel("eta")
    plt.ylabel("average accuracy")
    plt.xscale("log")
    plt.show()


def q1_b(train_data, train_labels, validation_data, validation_labels, C_arr):
    avg_acc = []

    for C in C_arr:
        C_acc = []
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, 1, 1000)
            calc = calc_successes(w, validation_data, validation_labels)  # Counting successful classifications
            C_acc.append(calc)

        avg_acc.append(np.mean(C_acc))

    plt.plot(C_arr, avg_acc)
    plt.xlabel("C")
    plt.ylabel("average accuracy")
    plt.xscale("log")
    plt.show()


def q1_c_and_d(train_data, train_labels, validation_data, validation_labels):
    # 1_c
    w = SGD_hinge(train_data, train_labels, 10**(-4), 1, 20000)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')  # Using the suggested function
    plt.show()

    # 1_d
    print("Accuracy of best classifier (hinge) on test set:")
    print(calc_successes(w, validation_data, validation_labels))


def q2_a(train_data, train_labels, validation_data, validation_labels, etas):
    avg_acc = []

    for eta in etas:
        eta_acc = []
        for i in range(10):
            w = SGD_log(train_data, train_labels, eta, 1000, False)
            calc = calc_successes(w, validation_data, validation_labels)  # Counting successful classification
            eta_acc.append(calc)

        avg_acc.append(np.mean(eta_acc))

    plt.plot(etas, avg_acc)
    plt.xlabel("eta")
    plt.ylabel("average accuracy'")
    plt.xscale("log")
    plt.show()


def q2_b_and_c(train_data, train_labels, validation_data, validation_labels):
    w = SGD_log(train_data, train_labels, 10**(-5), 20000, True)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')  # Using the suggested function
    plt.show()

    print("Accuracy of best classifier (log) on test set:")
    print(calc_successes(w, validation_data, validation_labels))


def calc_successes(w, validation_set, validation_labels_calc_acc):
    n = len(validation_set)
    correct = 0
    for i in range(n):
        x_i = validation_set[i]
        y_i = validation_labels_calc_acc[i]
        if numpy.sign(w @ numpy.transpose(x_i)) == y_i:
            correct += 1
    return correct / n


def calc_grad_llog(w,x,y):
    """
    grad(log(1+e^(-ywx)))= -y*e^(-yxw)/(1+e^(-yxw)) * x
    """
    a=(-y*(x@numpy.transpose(w)))
    grad=-y*(scipy.special.softmax(np.array([0,a]))[1])*x
    return grad


if __name__ == '__main__':
    res = helper()
    eta_arr = [10 ** i for i in range(-5, 4)]
    C_arr = [10 ** i for i in range(-5, 6)]

    # Q1
    q1_a(res[0], res[1], res[2], res[3], eta_arr)
    q1_b(res[0], res[1], res[2], res[3], C_arr)
    q1_c_and_d(res[0], res[1], res[2], res[3])

    # Q2
    q2_a(res[0], res[1], res[2], res[3], eta_arr)
    q2_b_and_c(res[0], res[1], res[2], res[3])



