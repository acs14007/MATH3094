import matplotlib.pyplot as plt
import numpy as np
import warnings
import copy


seed = 231
samples = 1000

np.random.seed(seed=seed)

def normalize_data(_data):
    _means = np.mean(_data, axis=0)
    _stds = np.std(_data, axis=0)
    _data = (_data - _means) / _stds
    return _data


def principle_component_analysis(_input_data, number_of_components: int):
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

    plt.scatter(feature1+1, feature2+1, color='m')
    plt.plot([feature1.min() * -weights[0][0], feature1.max() * -weights[0][0]], [feature1.min() * -weights[0][1], feature1.max() * -weights[0][1]], color='gold')
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
        print(f'PCA component {i} has {np.round(percentage_of_total_variance, 2)}% of total variance which' +
              f' explains {np.round(_cumulative_variance, 2)}% of total variance')
    return components


# Does distribution of data matter when applying PCA?

feature1 = 1 / np.random.gamma(1.0000001, 1, samples)
feature2 = 1 / np.random.gamma(1.0000001, 1, samples)
feature2 = feature1 * feature2
# feature2 = np.random.normal(size=samples)
data = np.vstack((feature1, feature2)).transpose()

pca = principle_component_analysis(data, 2)


samples = int(samples / 10)
feature1 = 1 / np.random.gamma(1.0000001, 1, samples)
feature2 = 1 / np.random.gamma(1.0000001, 1, samples)
feature2 = feature1 * feature2
data = np.vstack((feature1, feature2)).transpose()

pca = principle_component_analysis(data, 2)
