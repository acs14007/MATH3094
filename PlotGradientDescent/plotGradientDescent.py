import matplotlib
matplotlib.use("TkAgg")

import numpy as np
np.seterr(all='raise')
from matplotlib import pyplot as plt
from matplotlib import animation

import warnings


def mean_squared_error(actuals, *, predicted=None, predictors=None, linear_weights=None) -> float:
    """
    Calculates the Squared Error

    Either the predicted values or the model_inputs and weights need to be supplied
    :param actuals: A numpy ndarray of actual values
    :param predicted: A numpy ndarray of predictions
    :param predictors: A numpy ndarray of predictors
    :param linear_weights: The weights for a linear model for the predictors
    :return: The squared error
    """
    if predicted is not None:
        try:
            return np.square(predicted - actuals).average()
        except:
            warnings.warn('There was an error2')
            return np.inf
    elif (predictors is not None) and (linear_weights is not None):
        try:
            return np.average(np.square(actuals.flatten() - np.matmul(predictors.reshape(len(predictors), 2), linear_weights.flatten())))
        except:
            print(actuals, predictors, linear_weights)
            warnings.warn('There was an error1')
            return np.inf
    else:
        raise ValueError


def get_gradient_linear(actuals, predictors, linear_weights):
    """
    Calculates the gradient
    :return: The gradient
    """
    predicted = np.matmul(predictors, linear_weights)
    a = -2 * np.average(predictors[:, 0].flatten() * (actuals.flatten() - predicted.flatten()))
    b = -2 * np.average((actuals.flatten() - predicted.flatten()))
    return np.array([a, b])
    # try:
    #     return np.matmul(np.matmul(predictors.transpose(), predictors), linear_weights) - np.matmul(predictors.transpose(), actuals).flatten()
    # except ValueError:
    #     warnings.warn('There was an error')
    #     return linear_weights * 0


# Some info
filename = r'D:\MATH3094\data\auto-mpg\auto-mpg.csv'


# Import my data
data = np.loadtxt(filename, dtype=np.dtype('U40'), delimiter=',', skiprows=1)
_header = np.loadtxt(filename, delimiter=",", comments="\n", dtype=np.unicode_, max_rows=1)
_header = np.char.strip(_header, '"')
Xraw = data[:, [np.where(_header == 'displacement')[0][0]]].astype(np.dtype(np.longdouble))
Y = data[:, [np.where(_header == 'mpg')[0][0]]].astype(np.dtype(np.longdouble))
X = np.append(Xraw, np.ones([Xraw.shape[0], 1]), axis=1)



# Set up my figure
fig = plt.figure()
ax = plt.axes(xlim=(0, 500), ylim=(0, 50))
ax.set_xlabel('Engine Displacement')
ax.set_ylabel('Miles Per Gallon')
ax.scatter(Xraw, Y, color='m')
line, = ax.plot([], [], lw=2, color='gold')


def init():
    line.set_data([], [])
    return line,


step_size = np.array([0.00002, 0.01])
number_of_steps = 1500
model_weights = np.array([1, 0], dtype=np.dtype(np.longdouble))

iterations = []
MSEs = []
def animate(i):
    global model_weights
    model_weights -= step_size * get_gradient_linear(Y, X, model_weights)
    # print(model_weights)
    x = np.linspace(0, 500, 1000)
    y = ((float(model_weights[0]) * x) + float(model_weights[1]))
    # y = 25*np.sin(2 * np.pi * (x - 0.01 * i)) + 25
    if i % 100 == 0:
        print(i, model_weights)
    line.set_data(x, y)

    iterations.append(i)
    MSEs.append(mean_squared_error(Y.flatten(), linear_weights=model_weights, predictors=X))
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=number_of_steps, interval=4, blit=True)

# plt.show()
anim.save('animation.mp4', writer='ffmpeg', fps=60)
print('Done')


plt.clf()
# Now we plot the MSE vs. iteration number
fig = plt.figure()
ax = plt.axes(xlim=(0, np.max(iterations)), ylim=(0, np.log(1.1 * np.max(MSEs))))
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Logarithmic Mean Squared Error')
ax.plot(iterations, np.log(MSEs), color='mediumspringgreen')
plt.show()


# fig, (ax1, ax2) = plt.subplots(1, 2)