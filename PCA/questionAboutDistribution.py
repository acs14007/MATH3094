import matplotlib.pyplot as plt
import numpy as np
import warnings
import copy
import sys

sys.path.append(r'D:\MATH3094\LinearRegression')
from linearregression import LinearModel

seed = int(318)

np.random.seed(seed=seed)


def normalize_data(_data):
    _means = np.mean(_data, axis=0)
    _stds = np.std(_data, axis=0)
    _data = (_data - _means) / _stds
    return _data


def principle_component_analysis(_input_data, number_of_components: int, should_plot=False, return_first_amount=False):
    if number_of_components > _input_data.shape[1]:
        warnings.warn('Too many components')
        raise ValueError

    # Copy data
    _input_data = copy.deepcopy(_input_data)

    # Scale data
    _input_data = normalize_data(_input_data)
    _stds = np.std(_input_data, axis=0)

    # Covariance Matrix
    fast_covariance = True
    _covariance_matrix = np.nan
    if fast_covariance:
        _covariance_matrix = np.cov(_input_data, rowvar=False)
    if not fast_covariance:
        _covariance_matrix = np.matmul(_input_data.T, _input_data) / _input_data.shape[0]

    # Get the eigenvalues
    # eigenvalues, eigenvectors = np.linalg.eig(_covariance_matrix)
    eigenvectors, s, v = np.linalg.svd(_covariance_matrix)

    weights = eigenvectors[:, :number_of_components]

    feature1 = _input_data[:, 0]
    feature2 = _input_data[:, 1]

    if should_plot:
        plt.scatter(feature1 + 1, feature2 + 1, color='m')
        plt.plot([feature1.min() * -weights[0][0], feature1.max() * -weights[0][0]],
                 [feature1.min() * -weights[0][1], feature1.max() * -weights[0][1]], color='gold')
        plt.show()
        print(weights)

    # Total Variability
    total_variability = np.sum(np.diagonal(_covariance_matrix))

    # Make components
    components = np.matmul(weights.T, _input_data.T).T

    _cumulative_variance = 0
    for i in range(number_of_components):
        component = components[:, i]
        std = component.std()
        variance = std ** 2
        percentage_of_total_variance = variance * 100 / total_variability
        _cumulative_variance += percentage_of_total_variance
        if should_plot:
            print(f'PCA component {i} has {np.round(percentage_of_total_variance, 2)}% of total variance which' +
                  f' explains {np.round(_cumulative_variance, 2)}% of total variance')
        if return_first_amount:
            return (percentage_of_total_variance)
    return components, weights


# %%

# How do we know to use PCA?
samples = 1000
feature11 = 1 / np.random.gamma(1.0000001, 1, samples)
feature12 = np.random.normal(size=samples) * feature11

train_actuals = feature11 + feature12
train_data = np.vstack((feature11, feature12)).transpose()
print(train_data.shape)
pca, weights = principle_component_analysis(train_data, 2, True)
full_train_data = np.vstack((train_data.T, train_actuals.T)).T
np.savetxt(f'train{seed}.csv', full_train_data, delimiter=',', comments='', fmt='%s',
           header='feature1, feature2, target')

samples = int(samples / 10)
feature21 = 1 / np.random.gamma(1.0000001, 1, samples)
feature22 = np.random.normal(size=samples) * feature21

test_actuals = feature21 + feature22
test_data = np.vstack((feature21, feature22)).transpose()
pca, weights = principle_component_analysis(test_data, 2, True)
full_test_data = np.vstack((test_data.T, test_actuals.T)).T
np.savetxt(f'test{seed}.csv', full_test_data, delimiter=',', comments='', fmt='%s', header='feature1, feature2, target')


# %%

def mean_squared_error(*, actuals, predicted) -> float:
    return np.square(predicted - actuals).sum() / actuals.shape[0]


test_data = np.concatenate([test_data, np.ones(shape=(test_data.shape[0], 1))], axis=1)
train_data = np.concatenate([train_data, np.ones(shape=(train_data.shape[0], 1))], axis=1)

# First I make a model on full dataset with both features
model = LinearModel(train_data, train_actuals)
model.calculate_optimal_model()
predicted = model.predict(test_data)

plt.title('Predicted vs. Actuals on test set with full dataset')
plt.xlabel('Actuals')
plt.ylabel('Predicted')
plt.plot([test_actuals.min(), test_actuals.max()], [test_actuals.min(), test_actuals.max()], color='gold')
plt.scatter(test_actuals, predicted)
plt.show()
mse = mean_squared_error(actuals=test_actuals, predicted=predicted)
print(mse)
# So this model is good


# Now we make a model for the PCA versions;
# we only use component 1 since that has 99.32% of variance
train_data = train_data[:, 0:2]
pca1, weights1 = principle_component_analysis(train_data, number_of_components=1, should_plot=False)
pca1 = np.concatenate([pca1, np.ones(shape=(pca1.shape[0], 1))], axis=1)
model = LinearModel(pca1, train_actuals)
model.calculate_optimal_model()

plt.title('Predicted vs. Actuals on train set with 1 PCA component')
plt.xlabel('Actuals')
plt.ylabel('Predicted')
plt.plot([train_actuals.min(), train_actuals.max()], [train_actuals.min(), train_actuals.max()], color='gold')
plt.scatter(train_actuals, model.predict(pca1), color='m')
plt.show()
# It is not bad on the train set

# How does this model perform on the train set though?
# Get the PCA component1 for the test set
test_data = test_data[:, 0:2]
component = np.matmul(weights1.T, test_data.T).T
component = np.concatenate([component, np.ones(shape=(component.shape[0], 1))], axis=1)
predicted = model.predict(component)
mse = mean_squared_error(predicted=predicted, actuals=test_actuals)
print(mse)
# Very high MSE


plt.title('Predicted vs. Actuals on test set with 1 PCA component')
plt.xlabel('Actuals')
plt.ylabel('Predicted')
plt.plot([test_actuals.min(), test_actuals.max()], [test_actuals.min(), test_actuals.max()], color='gold')
plt.scatter(test_actuals, predicted, color='m')
plt.show()
















# %%


# for i in range(100000):
#     seed = int(i)
#     np.random.seed(seed=seed)
#     samples = 1000
#     feature11 = np.random.normal(size=samples)
#     feature11[feature11 >= 0] = 1
#     feature11[feature11 < 0] = 0
#     feature12 = np.random.normal(size=samples) * feature11
#     train_actuals = feature11 + feature12
#     train_data = np.vstack((feature11, feature12)).transpose()
#
#     pca1 = principle_component_analysis(train_data, 2, False, True)
#
#     samples = int(samples / 10)
#     feature21 = np.random.normal(size=samples)
#     feature21[feature21 >= 0] = 1
#     feature21[feature21 < 0] = 0
#     feature22 = np.random.normal(size=samples) * feature21
#
#     test_actuals = feature21 + feature22
#     test_data = np.vstack((feature21, feature22)).transpose()
#     pca2 = principle_component_analysis(test_data, 2, False, True)
#
#     if pca1 > 60:
#         print(pca1)
#         if pca2 < 53:
#             print(pca1, pca2, seed)

