import numpy as np
import PIL.Image as Image
import torch
import random
import colorsys
import pathlib

from torch.autograd import Variable
from PIL.ImageOps import invert
from torchvision.utils import make_grid
from typing import Union, Optional, List, Tuple, Text, BinaryIO
from skimage import color


def m_rgb_to_hsv(src):
    if isinstance(src, Image.Image):
        r, g, b = src.split()

        h_dat = []
        s_dat = []
        v_dat = []

        for rd, gn, bl in zip(r.getdata(), g.getdata(), b.getdata()):
            h, s, v = colorsys.rgb_to_hsv(rd / 255., gn / 255., bl / 255.)
            h_dat.append(int(h * 255.))
            s_dat.append(int(s * 255.))
            v_dat.append(int(v * 255.))
        r.putdata(h_dat)
        g.putdata(s_dat)
        b.putdata(v_dat)

        return Image.merge('RGB', (r, g, b))
    else:
        return None


def m_hsv_to_rgb(src):
    if isinstance(src, Image.Image):
        r, g, b = src.split()

        h_dat = []
        s_dat = []
        v_dat = []

        for rd, gn, bl in zip(r.getdata(), g.getdata(), b.getdata()):
            h, s, v = colorsys.hsv_to_rgb(rd/255., gn/255., bl/255.)
            h_dat.append(int(h*255.))
            s_dat.append(int(s*255.))
            v_dat.append(int(v*255.))
        r.putdata(h_dat)
        g.putdata(s_dat)
        b.putdata(v_dat)

        return Image.merge('RGB', (r, g, b))
    else:
        return None


def numpy_to_pil(src):
    return Image.fromarray(np.uint8(src), 'RGB')


def tensor_to_pil(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0):

    ndarr = tensor_to_numpy(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                            normalize=normalize, range=range, scale_each=scale_each)
    pil_im = Image.fromarray(ndarr)

    return pil_im


def tensor_to_numpy(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0):

    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    return ndarr


def rgb_to_lab(src):
    """
    :param src: numpy array, PIL
    :return: numpy array
    """

    return color.rgb2lab(src)


def lab_to_rgb(src):
    """
    :param src: numpy array, PIL
    :return: numpy array
    """

    return color.lab2rgb(src)


if __name__ == '__main__':
    img = Image.open('dataset/Lenna.png', 'r')
    img = rgb_to_lab(img)
    img_np_lab = np.array(img)
    print(img_np_lab)
