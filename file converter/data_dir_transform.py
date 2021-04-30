import os

image_dir = 'A:/temp/temp'
transfer_destination_1 = 'A:/temp/temp'
transfer_destination_2 = 'A:/temp/temp'
image_list = os.listdir(image_dir)

if not os.path.exists(transfer_destination_1):
    os.mkdir(transfer_destination_1)
if not os.path.exists(transfer_destination_2):
    os.mkdir(transfer_destination_2)


def window_join(dir1, dir2):
    return dir1 + '/' + dir2


for idx, image_name in enumerate(image_list):
    image_path = window_join(image_dir, image_name)
    f_read = open(image_path, 'rb')

    if image_path[-5] == 'a':       # condition 1
        f_write_input = open(window_join(transfer_destination_1, image_name), 'wb')
        f_write_input.write(f_read.read())
        f_write_input.close()

    elif image_path[-5] == 'b':     # condition 2
        image_name = list(image_name)
        image_name[-5] = 'a'    # both image has to be same name
        tmp = ''.join(image_name)

        f_write_output = open(window_join(transfer_destination_2, tmp), 'wb')
        f_write_output.write(f_read.read())
        f_write_output.close()

    elif image_path[-8] == 'b':     # condition 2
        image_name = list(image_name)
        image_name[-8] = 'a'    # both image has to be same name
        tmp = ''.join(image_name)

        f_write_output = open(window_join(transfer_destination_2, tmp), 'wb')
        f_write_output.write(f_read.read())
        f_write_output.close()
    else:
        print('Unexpected condition for', image_path)

    f_read.close()

print('Done!')
