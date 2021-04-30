import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gradient_right(x):
    """
    :param x: shape[batch_size, 1d_tensor]
    :return:
    """
    x = x.view([x.shape[0], -1])
    pad = nn.ConstantPad1d(1, 0)

    x_pad = pad(x)
    x_2 = x_pad[:, :-2]
    x_right = x - x_2

    return x_right[:, 1:]


def gaussian(window_size, sigma):
    """
    Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
    :param window_size: size of the SSIM sampling window e.g. 11
    :param sigma: Gaussian variance
    :returns: 1xWindow_size Tensor of Gaussian weights
    :rtype: Tensor
    """
    gauss = torch.Tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


C1 = 0.01 ** 2
C2 = 0.03 ** 2
C3 = C2 / 2

img1 = torch.tensor([1, 1, 1, 1, 1, 9, 9, 9, 9, 9]).unsqueeze(0).unsqueeze(0).float() / 1000
# img2 = torch.tensor([1, 1, 1, 1, 1, 9, 9, 9, 9, 9]).unsqueeze(0).unsqueeze(0).float() / 1000
# img2 = torch.tensor([3, 3, 3, 3, 3, 5, 5, 5, 5, 5]).unsqueeze(0).unsqueeze(0).float() / 1000
# img2 = torch.tensor([99, 99, 99, 99, 99, 1, 1, 1, 1, 1]).unsqueeze(0).unsqueeze(0).float() / 1000
img2 = torch.tensor([-54, -235, -6345, 234, 12, -76, 756, -54, 65, 4]).unsqueeze(0).unsqueeze(0).float() / 1000

# img1 = gradient_right(img1).unsqueeze(0)
# img2 = gradient_right(img2).unsqueeze(0)

_1D_window = gaussian(img1.shape[2], 1.5).unsqueeze(0).unsqueeze(0)

mu1 = F.conv1d(img1, _1D_window, padding=1 // 2, groups=1)
mu2 = F.conv1d(img2, _1D_window, padding=1 // 2, groups=1)

mu1_sq = mu1.pow(2)
mu2_sq = mu2.pow(2)
mu1_mu2 = mu1 * mu2

sigma1_sq = F.conv1d(img1 * img1, _1D_window, padding=1 // 2, groups=1) - mu1_sq
sigma2_sq = F.conv1d(img2 * img2, _1D_window, padding=1 // 2, groups=1) - mu2_sq
sigma12 = F.conv1d(img1 * img2, _1D_window, padding=1 // 2, groups=1) - mu1_mu2

sigma1_sq = max(0, sigma1_sq)
sigma2_sq = max(0, sigma2_sq)

sigma1 = math.sqrt(sigma1_sq)
sigma2 = math.sqrt(sigma2_sq)

l_error = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)     # Luminance error
c_error = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)   # Contrast error
s_error = (sigma12 + C3) / (sigma1 * sigma2 + C3)   # Structure error

print(l_error)  # 같으면 1, 달라질수록 음수 방향으로...
print(c_error)  # 같으면 1, 달라질수록 음수 방향으로...
print(s_error)  # 같으면 1, 달라질수록 음수 방향으로...
