"""
Based of of what I read here:
https://en.wikipedia.org/wiki/Tikhonov_regularization

Basically we introduce another term into linear regression to help prevent overfitting. Maybe more generalizable models
"""


import matplotlib.pyplot as plt
import numpy as np
import warnings

class TikhonovModel:
    def __init__(self, trainX, trainY, ridge_parameter, *, dtype=float):
        self.dtype = dtype
        self.trainX = np.array(trainX, dtype=self.dtype)
        self.trainY = np.array(trainY, dtype=self.dtype)
        self.ridge_parameter = float(ridge_parameter)

        # Check the data matches the same length
        if trainX.shape[0] != trainY.shape[0]:
            raise IndexError
        # Check that the last column is all 1's
        if (len(np.unique(trainX[:, -1])) != 1) or (trainX[0, -1] != 1):
            warnings.warn('The last column should be all ones')

        self.model_weights = np.array([1 for _ in range(trainX.shape[1])], dtype=np.dtype(self.dtype))

        # Set some more variables we may or may not use
        self.scale_factors = np.array([1 for _ in range(trainX.shape[1])], dtype=np.dtype(self.dtype))

    def scale_data(self):
        """
        Scales the data to make convergence faster
        """
        pass

    def _get_squared_error(self):
        """
        Returns the squared error of the current model
        :return:
        """
        return np.square(self.trainY - np.matmul(self.trainX, self.model_weights)).sum()

    def calculate_optimal_model(self):
        """
        Doing this by least squares can be very expensive if the data is large
        :return:
        """
        _D = np.matmul(self.trainX.transpose(), self.trainX) +\
             (self.ridge_parameter * np.identity(self.trainX.shape[1], dtype=self.dtype))
        self.model_weights = np.matmul(np.matmul(np.linalg.inv(_D), self.trainX.transpose()), self.trainY)
        return self.model_weights


    def predict(self, new_predictors):
        testX = np.array(new_predictors, dtype=self.dtype)
        if testX.shape[1] != self.trainX.shape[1]:
            raise ValueError
        if testX.shape[0] < 1:
            raise ValueError

        return(np.matmul(testX, self.model_weights))