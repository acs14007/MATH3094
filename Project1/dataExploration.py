from supportCode import *
from PIL import Image

import os


dtype = np.dtype(int)
default_color = 255
key_file_location = r'D:\MATH3094\Project1\Data\train.csv'
resized_images_data_folder = r'D:\MATH3094\Project1\Data\Rescaledimages'
resized_images_original_shape_file = r'D:\MATH3094\Project1\Data\Rescaledimages\train_meta.csv'



if __name__ == '__main__':
    train_folder = os.path.join(resized_images_data_folder, 'train')
    list_of_photos = os.listdir(train_folder)

    key_file_data, key_file_header = load_a_file(key_file_location)
    resized_images_shape_data, resized_images_shape_header = load_a_file(resized_images_original_shape_file)

    random_image_name = list_of_photos[np.random.randint(0, len(list_of_photos))]
    random_image_location = os.path.join(train_folder, random_image_name)
    random_image = Image.open(random_image_location)

    image_id = os.path.basename(random_image_location)[:-4]
    random_image_data = np.array(random_image, dtype=dtype)

    random_image_data = histogram_equalization(random_image_data)
    random_image_size = np.array(random_image_data.shape)
    random_image_original_size = resized_images_shape_data[resized_images_shape_data[:, 0] == image_id, 1:].astype(np.dtype(int))
    random_image_scale = np.mean(random_image_original_size / random_image_size)

    # Mark the middle
    current_middle = int(random_image_data.shape[1] / 2)

    # random_image_data = draw_vertical_line(random_image_data, current_middle)
    # random_image_data = draw_box(random_image_data, 200, 200, 250, 250)

    rows = key_file_data[key_file_data[:, 0] == image_id]
    for row in rows:
        if row[4] == '':
            pass
        else:
            original_values = row[-4:].astype(np.dtype(float))
            new_values = original_values / random_image_scale
            random_image_data = draw_box(random_image_data, list_of_values=new_values)
            print(row)

    Image.fromarray(random_image_data).show()

