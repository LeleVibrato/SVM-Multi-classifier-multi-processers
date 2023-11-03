# -*- coding: utf-8 -*-
# @Date: 2022-01-01
# @Author: Sam
# @Filename: nlsvm_ovo_poly_kernel_multi_process.py
# @Software: SVM Multi-classifier (OvO method, polynomial kernel, parallel computation)
# @License: MIT

import numpy as np
from scipy import sparse
import osqp
import multiprocessing as mp
from functools import partial

# The default is to use a polynomial kernel function. To use other kernel functions, modify the program code.


def poly_kernel(X1, X2, degree):
    """
    Polynomial kernel function.

    :param X1: The first vector.
    :param X2: The second vector.
    :param degree: The degree of the polynomial.
    :return: The inner product of the two vectors.
    :rtype: float
    """
    return (1 + np.dot(X1, X2)) ** degree

# Uncomment the following function to use a Gaussian kernel instead.
# def gaussian_kernel(X1, X2, sigma):
#     """
#     Gaussian kernel function.
#
#     :param X1: The first vector.
#     :param X2: The second vector.
#     :param sigma: The sigma parameter of the Gaussian kernel.
#     :return: The inner product of the two vectors.
#     :rtype: float
#     """
#     return np.exp(-np.linalg.norm(X1 - X2) ** 2 / (2 * (sigma ** 2)))


class NLSVM:
    def __init__(self, beta0, alpha, y, X, degree, categories):
        self.beta0 = beta0
        self.alpha = alpha
        self.y = y
        self.X = X
        self.n = self.X.shape[0]
        self.categories = categories
        self.degree = degree

    def __call__(self, x):
        result = self.beta0 + sum(self.alpha[i] * self.y[i, 0] * (
            1 + np.dot(x, self.X[i, :])) ** self.degree for i in range(self.n))
        return self.categories[0] if result >= 0.0 else self.categories[1]


def nlsvm_solve(model, index, total, X, y, categories, C, degree):
    """
    Trains a binary SVM classifier using a polynomial kernel function.

    :param model: A list of SVM classifiers.
    :param index: The current classifier count.
    :param total: The total number of classifiers.
    :param X: The feature matrix of the training data, n rows by dim columns, where dim is the number of dimensions, each row is a training sample, and each column is a feature.
    :param y: The label matrix of the training data, n rows by 1 column, values are either -1 or 1.
    :param categories: A tuple of two strings representing the binary classes used for training.
    :param C: The SVM hyperparameter, the upper bound for alpha_i.
    :param degree: The degree of the polynomial kernel function.
    :return: An SVM binary classifier.
    """
    n = X.shape[0]  # Number of samples
    # Calculate inner products
    inner_product = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1):
            inner_product[i, j] = inner_product[j, i] = poly_kernel(
                X[j, :], X[i, :], degree)

    # Define quadratic programming parameters matrix
    q = -np.ones(n)
    P = 0.5 * (y @ y.T) * inner_product
    P[np.diag_indices_from(P)] = 2 * P[np.diag_indices_from(P)]
    A = np.vstack((np.diag(np.ones(n)), y.T))
    lb = np.zeros(n + 1)
    ub = np.hstack((C * np.ones(n), [0.0]))
    # Solve the quadratic programming problem
    prob = osqp.OSQP()
    prob.setup(sparse.csc_matrix(P), q,
               sparse.csc_matrix(A), lb, ub, verbose=False)
    res = prob.solve()
    alpha = res.x

    # Find the indices of the support vectors
    svs_ind = np.where((alpha > 0.0) & (alpha < C))[0]
    # Compute beta0 (average)
    beta0sum = sum(1.0 / y[j, 0] - sum(alpha[i] * y[i, 0] *
                   inner_product[j, i] for i in range(n)) for j in svs_ind)
    beta0 = beta0sum / len(svs_ind)  # Take the average
    print(
        f"Classifier training completed: {index + 1}/{total} ['{categories[0]}', '{categories[1]}']")
    model.append(NLSVM(beta0, alpha, y, X, degree, categories))


def ovo(y):
    """
    Split the categories into multiple one-vs-one tuples.

    :param y: A one-dimensional list of strings, representing the classification of each training sample.
    :return: The binary categories for each classifier.
    :rtype: A list of 2-element tuples, where the tuple elements are strings.
    """
    unique_categories = np.unique(y)
    unique_categories.sort()
    number_of_categories = len(unique_categories)  # Number of categories
    print("Categories: ", unique_categories)
    return [(unique_categories[i], unique_categories[j])
            for i in range(number_of_categories) for j in range(i + 1, number_of_categories)]


def nlsvm(X, y, C=1.0, degree=3):
    """
    Interface function for model training.

    :param X: The feature matrix of training data, n rows by dim columns, where dim is the number of dimensions; each row is a training sample, and each column is a feature.
    :param y: The label list of training data, a one-dimensional numpy.array of strings.
    :param C: A floating-point number greater than 0, the SVM hyperparameter, the upper bound for alpha_i.
    :param degree: An integer greater than 1, the degree of the polynomial kernel function.
    :return: A multi-classifier for SVM.
    :rtype: A list composed of object functions, each representing a classifier.
    """
    n, dim = X.shape
    number_of_classes = len(np.unique(y))
    categories = ovo(y)  # Split into multiple one-vs-one tuples
    total_classifiers = len(categories)
    model = mp.Manager().list()
    process_pool = []
    print(
        f"A total of {len(categories)} classifiers are required for this task...")
    for i in range(len(categories)):
        class_A_data = X[y == categories[i][0], :]
        class_B_data = X[y == categories[i][1], :]
        y_class_A = np.ones((class_A_data.shape[0], 1))
        y_class_B = -np.ones((class_B_data.shape[0], 1))
        X_temp = np.vstack((class_A_data, class_B_data))
        y_temp = np.vstack((y_class_A, y_class_B))
        process_pool.append(mp.Process(target=nlsvm_solve, args=(
            model, i, total_classifiers, X_temp, y_temp, (categories[i][0], categories[i][1]), C, degree)))

    print("Starting processes...")
    for process in process_pool:
        process.start()
    for process in process_pool:
        process.join()
    print("Training complete!")
    print(
        f"Number of training samples: {n}; Dimensions: {dim}; Number of classes: {number_of_classes}")
    return model


def predict_1dim(model, x):
    """
    Predict the class for a single sample.

    :param model: The SVM multi-classifier, a list composed of function closures.
    :param x: 1-dimensional np.array.
    :return: The predicted class for x.
    :rtype: string
    """
    categories = list(map(lambda submodel: submodel(x), model))
    # Count occurrences for each class
    unique_class, counts = np.unique(categories, return_counts=True)
    # Return the class with the most occurrences
    return unique_class[np.argmax(counts)]


def predict(model, X):
    """
    Predict the classes for multiple samples.

    :param model: The SVM multi-classifier, a list composed of function closures.
    :param X: The feature matrix of the data to predict, np.array.
    :return: The predicted classes for each sample.
    :rtype: list of strings
    """
    num_samples = X.shape[0]  # Number of samples
    print(f"Number of test samples: {num_samples}")
    predict_func = partial(predict_1dim, model)
    pool_obj = mp.Pool()
    result = pool_obj.map(predict_func, X)
    return np.array(result)


def accuracy(ypredict, ytest):
    """
    Calculate the accuracy for the test samples.

    :param ypredict: One-dimensional array of predicted labels, type np.array(string).
    :param ytest: One-dimensional array of test labels, type np.array(string).
    :return: Accuracy.
    :rtype: A floating-point number between 0 and 1.
    """
    return sum(ypredict == ytest) / len(ypredict)


if __name__ == '__main__':
    from datetime import datetime
    import os
    from sklearn.model_selection import train_test_split
    import pandas as pd

    start_time = datetime.now()

    # Change directory to the script file location
    os.chdir(os.path.dirname(__file__))

    # Load training and testing data
    data_train = pd.read_csv(
        "zip.train", delimiter=" ", header=None).to_numpy()
    data_test = pd.read_csv("zip.test", delimiter=" ", header=None).to_numpy()
    X_train = data_train[:, 1:].astype(float)
    y_train = data_train[:, 0].astype(int).astype(str)
    X_test = data_test[:, 1:].astype(float)
    y_test = data_test[:, 0].astype(int).astype(str)

    print("Training on the handwritten digits dataset...")
    model = nlsvm(X_train, y_train, C=10.0, degree=6)
    print("Testing the classifier...")
    y_predict = predict(model, X_test)
    print(f"Test set accuracy: {accuracy(y_predict, y_test) * 100:.2f}%\n")

    end_time = datetime.now()
    print("Total time taken: ", (end_time - start_time).seconds, "seconds\n")
