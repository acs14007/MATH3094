from supportCode import *

from PIL import Image
import os
import cv2

dtype = np.dtype(int)
key_file_location = r'D:\MATH3094\Project1\Data\train.csv'
resized_images_data_folder = r'D:\MATH3094\Project1\Data\Rescaledimages'
resized_images_original_shape_file = r'D:\MATH3094\Project1\Data\Rescaledimages\train_meta.csv'

if __name__ == '__main__':
    train_folder = os.path.join(resized_images_data_folder, 'train')
    list_of_photos = os.listdir(train_folder)

    key_file_data, key_file_header = load_a_file(key_file_location)
    resized_images_shape_data, resized_images_shape_header = load_a_file(resized_images_original_shape_file)

    centers = []
    for _ in [1 for _ in range(10)]:
        image_name = list_of_photos[np.random.randint(0, len(list_of_photos))]
        image_location = os.path.join(train_folder, image_name)
        # image_location = r'D:\\MATH3094\\Project1\\Data\\Rescaledimages\\train\\5d092d091cf41b9291d00ff1286589cd.png'
        # image_location = r'D:\MATH3094\Project1\Data\Rescaledimages\train\015bf89fc34cde9fafe7c79366fecee7.png'
        # image_location = r'D:\\MATH3094\\Project1\\Data\\Rescaledimages\\train\\747eec98df4c54df053df178cc8c2395.png'
        # image_location = r'D:\MATH3094\Project1\Data\Rescaledimages\train\06aa6df528aa6ed39e732b186b38b915.png'
        # image_location = r'D:\MATH3094\Project1\Data\Rescaledimages\train\06a21aefd2f5e5ece6397e5fc10860b1.png'
        # image_location = r'D:\\MATH3094\\Project1\\Data\\Rescaledimages\\train\\f959f8e6ea3da784426e14ee78f849ac.png'
        image_id = os.path.basename(image_location)[:-4]

        rows = key_file_data[key_file_data[:, 0] == image_id]

        image = Image.open(image_location)
        image_data = np.array(image, dtype=dtype)

        image_data = histogram_equalization(image_data, False)

        try:
            center, spine_start, spine_end = find_spine(image_data, False)
            shoulders_index = find_shoulders(image_data, center, spine_start, spine_end, False)
            centers.append(center)
        except TypeError:
            spine_start, spine_end, center, shoulders_index = 0, 0, 0, 0
            print(image_location)


        body_start, body_end = find_sides_of_body(image_data, shoulders_index, center, spine_start, spine_end, False)



        # Show images
        # Image.fromarray(image_data).show()
        image_data = draw_vertical_line(image_data, spine_start)
        image_data = draw_vertical_line(image_data, spine_end)
        image_data = draw_vertical_line(image_data, center)
        image_data = draw_vertical_line(image_data, body_start)
        image_data = draw_vertical_line(image_data, body_end)
        image_data = draw_horizontal_line(image_data, shoulders_index)
        # Image.fromarray(image_data).show()
        # image_data = image_data[shoulders_index:, :]
        # image_data = image_data[body_start:body_end, int(shoulders_index):int(400 + shoulders_index)]
        image_data = image_data[int(shoulders_index):int(400 + shoulders_index), body_start:body_end]



        # image_data = image_data[int(-200 + center):int(200 + center), int(shoulders_index):int(400 + shoulders_index)]
        # si
        # image_data = np.pad(image_data, ((0, 400 - image_data.shape[0]), (0, 400 - image_data.shape[1])), mode='constant', constant_values=0)

        image_data = cv2.resize(image_data, (224, 224))
        plt.imshow(image_data, cmap='gray')
        plt.show()
    # plt.hist(centers, bins=40)
    # plt.title(f'Actual Centers Based on Spines')
    # plt.show()
