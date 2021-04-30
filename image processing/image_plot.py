import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import os
import math

plt.axis('off')
figure_dpi = 300

# limit the maximum row to avoid 'pyplot plot maximum size exception'.
PLT_ROW = 30


directories = [
    'A:/temp/temp',
    'A:/temp/temp',
    'A:/temp/temp',
]


PLT_COL = len(directories)

directory_list = list()
dir_len = 0
dir_len_past = 0
for i, item in enumerate(directories):
    file_names = os.listdir(item)
    dir_len = len(file_names)

    if i != 0:
        assert dir_len == dir_len_past, 'File length should be same: {:s}'.format(item)

    directory_list.append(file_names)
    dir_len_past = dir_len

image_show_row = len(directory_list[0])
# image_show_row = 10   # fix the the number of image

directory_list = np.array(directory_list).transpose().tolist()

render_iteration = 1
if image_show_row > PLT_ROW:
    render_iteration = math.ceil(image_show_row / PLT_ROW)

for render_idx in range(render_iteration):
    last_index = render_idx * PLT_ROW

    figure_width = 300 / figure_dpi    # width will be automatically resized by subplot
    figure_height = 1920 / figure_dpi

    if not render_idx + 1 == render_iteration:
        plt_render_images = directory_list[last_index:PLT_ROW * (render_idx + 1)]

        fig = plt.figure(figsize=(figure_width * PLT_COL, figure_height * PLT_ROW))
        fig.dpi = figure_dpi
    else:
        plt_render_images = directory_list[last_index:image_show_row]

        _, lefts = divmod(image_show_row, PLT_ROW)
        if lefts == 0:
            lefts = PLT_ROW
        fig = plt.figure(figsize=(figure_width * PLT_COL, figure_height * lefts))
        fig.dpi = figure_dpi

    for idx, image_zip in enumerate(plt_render_images):
        img_dr = list()
        for i in range(len(directories)):
            img_dr.append(os.path.join(directories[i], image_zip[i]))

        img_list = list()
        for item in img_dr:
            img_list.append(pil.open(item))

        for i in range((PLT_COL-1), -1, -1):
            ax = fig.add_subplot(image_show_row, PLT_COL, (idx+1) * PLT_COL - i)

            ax.imshow(img_list[(PLT_COL-1) - i])
            ax.axis('off')

        print('plotting {}th row...'.format((render_idx * PLT_ROW) + (idx + 1)))

    print('\nmaking image... wait for seconds')

    if not os.path.exists('./image_plot'):
        os.mkdir('./image_plot')

    fn = './image_plot/output_' + str(render_idx).zfill(3) + '.jpg'
    fig.savefig(fn, dpi=fig.dpi, bbox_inches='tight')
    print('saved', fn, end='\n\n')

print('Done!!!')
