import PIL.Image as Image
import os
import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from skimage import color


# input_dir = 'dataset/T4-41_a.jpg'
# output_dir = 'dataset/T4-41_out.jpg'
# target_dir = 'dataset/T4-41_b.jpg'
input_dir = 'dataset/T5-227-re_a.jpg'
output_dir = 'dataset/T5-227-re_out.jpg'
target_dir = 'dataset/T5-227-re_b.jpg'


def show_histogram(_input, _output, _target, channel, save_image=False, bins=100):
    """
    :param _input: Numpy Array
    :param _output: Numpy Array
    :param _target: Numpy Array
    :param channel: This variable decides the loop iterations. Choice of ['RGB', 'HSV', 'Lab'].
    :param save_image: Boolean
    :param bins: The number of 'x axis' in plot
    """

    for i in range(len(channel)):

        x_hist_dim = np.histogram(_input[:, :, i], bins=bins)[0]
        output_hist_dim = np.histogram(_output[:, :, i], bins=bins)[0]
        target_hist_dim = np.histogram(_target[:, :, i], bins=bins)[0]

        plt.plot(x_hist_dim, label=channel[i] + '_input')
        plt.plot(output_hist_dim, label=channel[i] + '_output')
        plt.plot(target_hist_dim, label=channel[i] + '_target')

        plt.legend()
        if save_image:
            if not os.path.exists('show_histogram'):
                os.mkdir('show_histogram')
            plt.savefig('show_histogram/' + channel + '_' + channel[i])
        # plt.show()
        plt.clf()

        print(x_hist_dim.sum())


def get_histogram(_input, _output, _target, channel='abc', bins=100):
    """
    :param _output: Tensor
    :param _target: Tensor
    :param channel_num: The number of channel
    :param bins: The number of 'x axis' in plot
    TODO: is this need _input Tensor?
    """

    # https://www.nature.com/articles/ncomms13890
    class GaussianHistogram(nn.Module):
        def __init__(self, bins, min, max, sigma):
            super(GaussianHistogram, self).__init__()
            self.bins = bins
            self.min = min
            self.max = max
            self.sigma = sigma
            self.delta = float(max - min) / float(bins)
            self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

        def forward(self, x):
            x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
            x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
            x = x.sum(dim=1)
            return x

    input_hist = torch.zeros([3, 100])

    for i in range(len(channel)):
        input_hist_dim_tensor = torch.histc(_input.cpu()[:, :, i], bins=bins).numpy()
        output_hist_dim_tensor = torch.histc(_output.cpu()[:, :, i], bins=bins).numpy()
        target_hist_dim_tensor = torch.histc(_target.cpu()[:, :, i], bins=bins).numpy()

        gh_input = GaussianHistogram(bins=100, min=_input.min(), max=_input.max(), sigma=1)
        input_hist[i] = gh_input(_input[:, :, i].view(-1))

        plt.plot(input_hist[i].numpy(), label=channel[i] + '_input')
        # plt.plot(output_hist_dim_tensor, label=channel[i] + '_output')
        # plt.plot(target_hist_dim_tensor, label=channel[i] + '_target')

        # print(input_hist_dim_tensor)
        # print(input_hist)
        #
        # exit(0)

        plt.legend()
        if not os.path.exists('show_histogram'):
            os.mkdir('show_histogram')
        plt.savefig('show_histogram/' + channel + '_' + channel[i])
        # plt.show()
        plt.clf()

    # return output_hist_dim_tensor, target_hist_dim_tensor


def main():

    x_rgb = np.array(Image.open(input_dir).convert('RGB'))
    output_rgb = np.array(Image.open(output_dir).convert('RGB'))
    target_rgb = np.array(Image.open(target_dir).convert('RGB'))

    x_hsv = color.rgb2hsv(x_rgb)
    output_hsv = color.rgb2hsv(output_rgb)
    target_hsv = color.rgb2hsv(target_rgb)

    x_lab = color.rgb2lab(x_rgb)
    output_lab = color.rgb2lab(output_rgb)
    target_lab = color.rgb2lab(target_rgb)

    input_lab_t = torch.Tensor(x_lab)
    output_lab_t = torch.Tensor(output_lab)
    target_lab_t = torch.Tensor(target_lab)

    get_histogram(input_lab_t, output_lab_t, target_lab_t)
    # get_histogram(output_rgb, target_rgb)

    # show_histogram(x_rgb, output_rgb, target_rgb, channel='HSV', save_image=True)
    # show_histogram(x_hsv, output_hsv, target_hsv, channel='HSV', save_image=True)
    # show_histogram(x_lab, output_lab, target_lab, channel='def', save_image=True)


if __name__ == '__main__':
    main()
