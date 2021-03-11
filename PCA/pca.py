# Written by Aaron Spaulding
# 3/10/2021


import numpy as np
import warnings
import copy


def normalize_data(_data):
    _means = np.mean(_data, axis=0)
    _stds = np.std(_data, axis=0)
    _data = (_data - _means) / _stds
    return(_data)


def principle_component_analysis(_input_data, number_of_components: int):
    if number_of_components > _input_data.shape[1]:
        warnings.warn('Too many components')
        raise ValueError

    # Copy data
    _input_data = copy.deepcopy(_input_data)

    # Scale data
    # _input_data = normalize_data(_input_data)
    _stds = np.std(_input_data, axis=0)

    # Covariance Matrix
    fast_covariance = True
    _covariance_matrix = np.nan
    if fast_covariance:
        _covariance_matrix = np.cov(_input_data, rowvar=False)
    if not fast_covariance:
        _covariance_matrix = np.matmul(_input_data.T, _input_data) / _input_data.shape[0]

    # Eigenvalues
    # eigenvalues, eigenvectors = np.linalg.eig(_covariance_matrix)
    u, s, v = np.linalg.svd(_covariance_matrix)
    eigenvectors = u

    weights = eigenvectors[:, :number_of_components]

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


if __name__ == '__main__':
    # Import our data
    filepath = r'D:\MATH3094\data\auto-mpg\auto-mpg.csv'
    data = np.loadtxt(filepath, dtype=np.dtype('U40'), delimiter=',', skiprows=1)
    header = np.loadtxt(filepath, delimiter=",", comments="\n", dtype=np.unicode_, max_rows=1)
    header = np.char.strip(header, '"')

    input_data = data[:, [np.where(header == 'cylinders')[0][0],
                          np.where(header == 'displacement')[0][0],
                          np.where(header == 'weight')[0][0],
                          np.where(header == 'acceleration')[0][0],
                          np.where(header == 'model year')[0][0]
                          ]].astype(np.dtype(float))




    # Now we want to check something
    #
    # I calculate the total variance of the data
    # The two statements below are the same, I use the second here and the first in my principle_component_analysis()
    #       total_variability = np.sum(np.diagonal(_covariance_matrix))
    #       total_variability = np.sum(_stds ** 2)
    #
    # Then I group the data into two groups
    #   1. The first three components: cylinders, displacement, and weight
    #   2. The last two components: acceleration, model year
    # I perform PCA on these two groups, the goal is to see how the total variance of each group of variables changes
    # It turns out that total variance of each of these sets sum to the total variance


    # input_data = normalize_data(input_data)
    stds_of_input_data = np.std(input_data, axis=0)
    variances_of_each_feature = stds_of_input_data ** 2
    total_variance_of_input_data = np.sum(variances_of_each_feature)
    print(variances_of_each_feature)
    print(total_variance_of_input_data)

    set1 = principle_component_analysis(input_data[:, :3], 3)
    # set1 = normalize_data(set1)
    stds_set1 = np.std(set1, axis=0)
    variances_features_set1 = stds_set1 ** 2
    total_variance_set1 = np.sum(variances_features_set1)
    print(variances_features_set1)
    print(total_variance_set1)

    set2 = principle_component_analysis(input_data[:, 3:], 2)
    # set2 = normalize_data(set2)
    stds_set2 = np.std(set2, axis=0)
    variances_features_set2 = stds_set2 ** 2
    total_variance_set2 = np.sum(variances_features_set2)
    print(variances_features_set2)
    print(total_variance_set2)
