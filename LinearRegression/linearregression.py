import matplotlib.pyplot as plt
import numpy as np
import warnings

np.seterr(all='raise')


class LinearModel:
    def __init__(self, trainX, trainY, *, dtype=float):
        self.dtype = dtype
        self.trainX = np.array(trainX, dtype=self.dtype)
        self.trainY = np.array(trainY, dtype=self.dtype)

        # Check the data matches the same length
        if trainX.shape[0] != trainY.shape[0]:
            raise IndexError
        # Check that the last column is all 1's
        if (len(np.unique(trainX[:, -1])) != 1) or (trainX[0, -1] != 1):
            warnings.warn('The last column should be all ones')

        self.model_weights = np.array([1 for _ in range(trainX.shape[1])], dtype=np.dtype(self.dtype))

        # Set some more variables we may or may not use
        self.scale_factors = np.array([1 for _ in range(trainX.shape[1])], dtype=np.dtype(self.dtype))
        self.step_size = np.array([0.00001 for _ in range(trainX.shape[1])], dtype=np.dtype(self.dtype))
        self.list_of_errors = np.empty(shape=(0, 2), dtype=self.dtype)

    def scale_data(self):
        """
        Scales the data to make convergence faster
        """
        pass

    def _get_gradient(self):
        return 2 * (np.matmul(np.matmul(self.trainX.transpose(), self.trainX), self.model_weights)
                    - np.matmul(self.trainX.transpose(), self.trainY))

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
        _D = np.matmul(self.trainX.transpose(), self.trainX)
        self.model_weights = np.matmul(np.matmul(np.linalg.inv(_D), self.trainX.transpose()), self.trainY)
        return self.model_weights

    def gradient_descent(self, iterations: int, *, step_size=None, verbose=False, should_plot=False):
        """
        This is preferred when we have large datasets
        :param iterations:
        :return:
        """
        if step_size is not None:
            self.step_size = step_size
        self.list_of_errors = np.empty(shape=(iterations, 2), dtype=self.dtype)
        for i in range(iterations):
            self.model_weights -= self.step_size * self._get_gradient()
            self.list_of_errors[i, 0] = i
            self.list_of_errors[i, 1] = self._get_squared_error()
            if verbose:
                print(i, self.model_weights, self.list_of_errors[i, 1], sep='\t\t')
        if should_plot:
            plt.clf()
            fig = plt.figure()
            ax = plt.axes(xlim=(0, iterations), ylim=(0, 1.1 * np.max(self.list_of_errors[:, 1])))
            ax.set_xlabel('Number of Iterations')
            ax.set_ylabel('Mean Squared Error')
            ax.plot(self.list_of_errors[:, 0], self.list_of_errors[:, 1], color='mediumspringgreen')
            plt.show()

    def predict(self, new_predictors):
        testX = np.array(new_predictors, dtype=self.dtype)
        if testX.shape[1] != self.trainX.shape[1]:
            raise ValueError
        if testX.shape[0] < 1:
            raise ValueError

        return(np.matmul(testX, self.model_weights))


if __name__ == '__main__':
    filename = r'D:\MATH3094\data\multivar_simulated\data.csv'
    data = np.genfromtxt(filename, skip_header=1, delimiter=',')
    Y = data[:, 1]
    X1 = data[:, 2:]
    X = np.concatenate([X1, np.ones(shape=(X1.shape[0], 1))], axis=1)

    model = LinearModel(X, Y)
