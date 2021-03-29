import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage

import copy
import warnings



def load_a_file(file_location):
    _data = np.loadtxt(file_location, delimiter=",", skiprows=1, dtype=np.unicode_)
    _data = np.char.strip(_data, '"')
    _header = np.loadtxt(file_location, delimiter=",", comments="\n", dtype=np.unicode_, max_rows=1)
    _header = np.char.strip(_header, '"')
    return _data, _header


def draw_box(image_data, x_min: int = None, y_min: int = None, x_max: int = None, y_max: int = None, *,
             list_of_values: list = None, color=255):
    if list_of_values is not None:
        x_min = int(list_of_values[0])
        y_min = int(list_of_values[1])
        x_max = int(list_of_values[2])
        y_max = int(list_of_values[3])
    if x_max >= image_data.shape[1]:
        warnings.warn('Box extends outside image border')
        x_max = image_data.shape[1] - 1
    if y_max >= image_data.shape[0]:
        warnings.warn('Box extends outside image border')
        y_max = image_data.shape[0] - 1
    if x_min < 0:
        warnings.warn('Box extends outside image border')
        x_min = 0
    if y_min < 0:
        warnings.warn('Box extends outside image border')
        y_min = 0
    image_data[y_min:y_max, x_min] = color
    image_data[y_min:y_max, x_max] = color
    image_data[y_min, x_min:x_max] = color
    image_data[y_max, x_min:x_max] = color
    return image_data


def histogram_equalization(image_data, should_plot=False):
    if type(image_data) is not np.ndarray:
        warnings.warn('Should be a numpy array')
    shape = image_data.shape
    _histogram, _bins = np.histogram(image_data.flatten(), 256, density=True)
    cdf = _histogram.cumsum()
    if should_plot:
        plt.plot(cdf)
        plt.show()
    # Now we normalize the cdf
    cdf = 255 * cdf / cdf[-1]
    image_data = np.interp(image_data.flatten(), _bins[:-1], cdf)
    return image_data.reshape(shape)


def histogram_s_curve(image_data):
    image_data = histogram_equalization(image_data, False)
    shape = image_data.shape
    _histogram, _bins = np.histogram(image_data.flatten(), 256, density=True)
    cdf = 1 / (1 + np.exp((1 / 25) * (158 - _bins)))
    # print('hiu', new_curve.max())
    # cdf = new_curve.cumsum()
    # plt.plot(cdf)
    # plt.show()
    cdf = 255 * cdf / cdf[-1]
    image_data = np.interp(image_data.flatten(), _bins, cdf)
    return image_data.reshape(shape)


def draw_vertical_line(image_data, index: int, /, *, color=255):
    if type(image_data) is not np.ndarray:
        warnings.warn('Should be a numpy array')
    image_data[:, index] = color
    return image_data


def draw_horizontal_line(image_data, index: int, /, *, color=255):
    if type(image_data) is not np.ndarray:
        warnings.warn('Should be a numpy array')
    image_data[index, :] = color
    return image_data


def find_spine(image_data, should_plot=False):
    smoothness = 50  # pixels
    width_of_area_to_check = 300  # pixels
    interval = 1  # pixels

    start_indexes = np.arange(int((image_data.shape[1] - width_of_area_to_check) / 2),
                              int((image_data.shape[1] + width_of_area_to_check) / 2), interval)
    end_indexes = start_indexes + smoothness
    energies = start_indexes * 0
    for i in range(start_indexes.shape[0]):
        start_index = start_indexes[i]
        end_index = end_indexes[i]
        energy = np.sum(np.square(image_data[:, start_index:end_index]))
        energies[i] = energy

    if should_plot:
        plt.plot(start_indexes, energies)
        plt.show()

    maximum_index, info = signal.find_peaks(energies, width=10, rel_height=0.5, distance=len(energies))
    spine_start = start_indexes[maximum_index]
    spine_end = end_indexes[maximum_index]
    center = int((spine_start + spine_end) / 2)
    spine_start = int(center - (info['widths'] / 2))
    spine_end = int(center + (info['widths'] / 2))
    return center, spine_start, spine_end


def find_shoulders(image_data, center=None, spine_start=None, spine_end=None, should_plot=False):
    min_height_of_area_to_check = 0
    max_height_of_area_to_check = 125
    threshold = 64
    sigma = 1
    buffer_to_ignore = 10

    if (center is None) or (spine_start is None) or (spine_end is None):
        warnings.warn('Information not provided so find_spine may be running twice')
        center, spine_start, spine_end = find_spine(image_data, False)
    _image_data = copy.deepcopy(image_data)
    _image_data[_image_data <= threshold] = 0
    _image_data[_image_data > threshold] = 255
    # Gaussian filter this
    _image_data = ndimage.gaussian_filter(_image_data, sigma=sigma)
    _image_data = np.transpose(ndimage.prewitt(np.transpose(_image_data)))
    # plt.imshow(_image_data, cmap='gray')
    # plt.show()
    _image_data = np.concatenate((_image_data[:, :int(spine_start - buffer_to_ignore)],
                                  _image_data[:, int(spine_end + buffer_to_ignore):]), axis=1)
    energies = np.sum(_image_data, axis=1)
    if should_plot:
        plt.plot(energies)
        plt.show()

    maximum_index, info = signal.find_peaks(energies[min_height_of_area_to_check:max_height_of_area_to_check],
                                            distance=max_height_of_area_to_check)
    return maximum_index[0]


def find_sides_of_body(image_data, shoulders_index=None, center=None, spine_start=None, spine_end=None,
                       should_plot=False):
    width_to_check_from_edge = 100
    if (center is None) or (spine_start is None) or (spine_end is None):
        warnings.warn('Information not provided so find_spine may be running twice')
        center, spine_start, spine_end = find_spine(image_data, should_plot)
    image_center = np.round(image_data.shape[0] / 2)
    if shoulders_index is None:
        warnings.warn('Information not provided so find_shoulders may be running twice')
        shoulders_index = find_shoulders(image_data, center, spine_start, spine_end, should_plot)
    _image_data = copy.deepcopy(image_data)
    _image_data = _image_data[shoulders_index:300, :]
    _image_data = histogram_s_curve(_image_data)

    energies = np.sum(_image_data, axis=0)
    energies = np.convolve(energies, [1 for _ in range(20)], mode='same')
    maximum_index, info = signal.find_peaks(energies, distance=125)
    if should_plot:
        x1 = np.arange(-1 * center, image_data.shape[1] - center)
        plt.plot(x1, energies)
        plt.show()
    try:
        return maximum_index[0], maximum_index[2]
    except IndexError:
        warnings.warn('An error occurred')
        return 0, image_data.shape[1] - 1
