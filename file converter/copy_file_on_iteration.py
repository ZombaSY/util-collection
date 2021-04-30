import os

# parameters
image_dir = 'A:/temp/temp'
copy_destination = 'A:/temp/temp'
copy_iteration = 10

image_list = os.listdir(image_dir)
origin_len = len(image_list)
file_iterator = 1

if not os.path.exists(copy_destination):
    os.mkdir(copy_destination)

for _ in range(copy_iteration):
    for idx, image_name in enumerate(image_list):
        image_path = os.path.join(image_dir, image_name)
        _, file_name_with_extension = os.path.split(image_path)
        file_name, file_extension = os.path.splitext(file_name_with_extension)

        f_read = open(image_path, 'rb')

        file_name = file_name[:4]   # special condition on file name
        new_file_name = file_name + '_' + str(file_iterator).zfill(5) + file_extension
        f_write = open(os.path.join(copy_destination, new_file_name), 'wb')
        f_write.write(f_read.read())
        f_write.close()

        file_iterator = file_iterator + 1


