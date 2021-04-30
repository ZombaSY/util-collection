import os

image_dir = 'A:/temp/temp'
image_list = os.listdir(image_dir)


def window_join(dir1, dir2):
    return dir1 + '/' + dir2


for idx, image_name in enumerate(image_list):
    image_path = window_join(image_dir, image_name)

    if image_path[-5] == 'A':       # condition 1
        os.remove(image_path)

    elif image_path[-5] == 'B':     # condition 2
        pass
    else:
        print('Unexpected condition for', image_path)


print('Done!')
