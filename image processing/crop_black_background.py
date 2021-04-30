import os

from PIL import Image
from PIL.ImageOps import invert


import numpy as np


def convert_handmade_src(src_path, output_size=None):

    def __crop_background(numpy_src):

        def __get_vertex(img):
            index = 0
            for i, items in enumerate(img):
                if items.max() != 0:  # activate where background is '0'
                    index = i
                    break

            return index

        numpy_src_y1 = __get_vertex(numpy_src)
        numpy_src_y2 = len(numpy_src) - __get_vertex(np.flip(numpy_src, 0))
        numpy_src_x1 = __get_vertex(np.transpose(numpy_src))
        numpy_src_x2 = len(numpy_src[0]) - __get_vertex(np.flip(np.transpose(numpy_src), 0))

        return numpy_src_x1, numpy_src_y1, numpy_src_x2, numpy_src_y2

    src_image = Image.open(src_path, 'r').convert('L')
    # src_image = invert(src_image)  # Invert image color

    numpy_image = np.asarray(src_image.getdata(), dtype=np.float64).reshape((src_image.size[1], src_image.size[0]))
    numpy_image = np.asarray(numpy_image, dtype=np.uint8)  # if values still in range 0-255

    pil_image = Image.fromarray(numpy_image, mode='L')
    x1, y1, x2, y2 = __crop_background(numpy_image)
    pil_image = pil_image.crop((x1, y1, x2, y2))

    if output_size is not None:
        pil_image = pil_image.resize([output_size, output_size])

    return pil_image


# Parameter
DATASET_PATH = 'dataset/'


def main():

    file_list = os.listdir(DATASET_PATH)

    for idx, fn in enumerate(file_list):
        img_path = DATASET_PATH + fn
        saved_path = 'crop_black_background/'

        cropped_image = convert_handmade_src(img_path)

        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
        cropped_image.save(saved_path + fn)
        print(saved_path + fn + '\t  saved!')


if __name__ == '__main__':
    main()
