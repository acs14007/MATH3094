from supportCode import *

from PIL import Image
import os
from functools import partial

types = [str(i) for i in range(14)]

dtype = np.dtype(int)
key_file_location = r'D:\MATH3094\Project1\Data\train.csv'
resized_images_data_folder = r'D:\MATH3094\Project1\Data\Rescaledimages'
resized_images_original_shape_file = r'D:\MATH3094\Project1\Data\Rescaledimages\train_meta.csv'


def get_the_things(_resized_images_shape_data, _image_location):
    image = Image.open(_image_location)
    width, height = image.size
    image_size = [height, width]

    image_original_size = _resized_images_shape_data[_resized_images_shape_data[:, 0] == image_id, 1:].astype(
        np.dtype(int))
    image_scale = np.mean(image_original_size / image_size)

    image_data = np.array(image, dtype=dtype)
    center, spine_start, spine_end = find_spine(image_data, False)
    shoulders_index = find_shoulders(image_data, center, spine_start, spine_end)

    original_values = row[-4:].astype(np.dtype(float))
    new_values = original_values / image_scale
    # image_data = draw_box(image_data, list_of_values=new_values)
    # Image.fromarray(image_data).show()
    # print(new_values)
    _y_min = new_values[1] - shoulders_index
    _y_max = new_values[3] - shoulders_index
    _x_min = new_values[0] - center
    _x_max = new_values[2] - center
    _y_middle = np.mean([_y_min, _y_max])
    _x_middle = int(np.mean([_x_min, _x_max]))
    if _x_min < 500:
        image_data = histogram_equalization(image_data)
        image_data = draw_vertical_line(image_data, spine_start)
        image_data = draw_vertical_line(image_data, spine_end)
        image_data = draw_vertical_line(image_data, center)
        image_data = draw_horizontal_line(image_data, shoulders_index)
        print(_x_min, _y_min, _x_max, _y_max)
        draw_box(image_data, list_of_values=new_values)
        draw_box(image_data, list_of_values=[-200 + center, 0 + shoulders_index, 0 + center, 450 + shoulders_index])
        draw_box(image_data, list_of_values=[0 + center, 0 + shoulders_index, 200 + center, 450 + shoulders_index])
        Image.fromarray(image_data).show()
        #
        # plt.show()
    return _y_middle, _x_middle, _x_min, _y_min, _x_max, _y_max


def plot_histogram(data, label1, label2, bins=40, min_x=None, max_x=None):
    plt.clf()
    if (min_x is not None) and (max_x is not None):
        plt.xlim(min_x, max_x)
    plt.hist(data, bins=bins)
    plt.title(f'{label1} of Bounding Box for finding {label2}')
    plt.savefig(f'type_{label2}_{label1}')
    plt.clf()


if __name__ == '__main__':
    train_folder = os.path.join(resized_images_data_folder, 'train')
    list_of_photos = os.listdir(train_folder)

    key_file_data, key_file_header = load_a_file(key_file_location)
    resized_images_shape_data, resized_images_shape_header = load_a_file(resized_images_original_shape_file)

    print(len(list_of_photos))
    types = ['1']
    for damage_type in types:
        y_middles = []
        x_middles = []
        x_mins = []
        x_maxes = []
        y_mins = []
        y_maxes = []
        i = 0
        for image_name in list_of_photos:
            print(i);
            i += 1
            image_location = os.path.join(train_folder, image_name)
            image_id = os.path.basename(image_location)[:-4]
            rows = key_file_data[key_file_data[:, 0] == image_id]
            for row in rows:
                if row[4] == '':
                    pass
                elif row[2] == damage_type:
                    func = partial(get_the_things, resized_images_shape_data)
                    y_middle, x_middle, x_min, y_min, x_max, y_max = func(image_location)

                    y_middles.append(y_middle)
                    x_middles.append(x_middle)
                    x_mins.append(x_min)
                    x_maxes.append(x_max)
                    y_mins.append(y_min)
                    y_maxes.append(y_max)
                else:
                    pass

        plot_histogram(x_mins, 'x_mins', str(damage_type), min_x=-256, max_x=256)
        plot_histogram(x_maxes, 'x_maxes', str(damage_type), min_x=-256, max_x=256)
        plot_histogram(y_mins, 'y_mins', str(damage_type), min_x=-0, max_x=512)
        plot_histogram(y_maxes, 'y_maxes', str(damage_type), min_x=-0, max_x=512)
        # plt.hist(y_middles, bins=40)
        # plt.xlim(0, 512)
        # plt.title(f'Center(y) of Bounding Box for finding {type}')
        # plt.savefig(f'type_{type}_y')
        # plt.clf()
        #
        # plt.hist(x_middles, bins=40)
        # plt.xlim(-256, 256)
        # plt.title(f'Center(x) of Bounding Box for finding {type}')
        # plt.savefig(f'type_{type}_x')
        # plt.clf()
