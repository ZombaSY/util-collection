import os

file_path = 'A:/temp/temp'

path_list = os.listdir(file_path)

with open('make_file_path.txt', 'wb') as f:
    for path in path_list:
        f.write(path.encode())  # 엔터키 안됨
        f.write(b'\n')
