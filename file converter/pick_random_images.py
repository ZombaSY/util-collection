import os
import random

pick_nums = 100
remove_refer = True

image_dir = 'A:/temp/temp'
transfer_destination_1 = 'A:/temp/temp'
image_list = os.listdir(image_dir)
random.shuffle(image_list)
image_list = image_list[:pick_nums]

if not os.path.exists(transfer_destination_1):
    os.mkdir(transfer_destination_1)


def window_join(dir1, dir2):
    return dir1 + '/' + dir2


for idx, image_name in enumerate(image_list):
    image_path = window_join(image_dir, image_name)
    f_read = open(image_path, 'rb')

    f_write_input = open(window_join(transfer_destination_1, image_name), 'wb')
    f_write_input.write(f_read.read())
    f_write_input.close()

    f_read.close()

    if remove_refer:
        os.remove(image_path)

print('Done!')
