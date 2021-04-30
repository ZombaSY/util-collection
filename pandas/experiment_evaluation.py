import torch
import torch.nn.functional as F
import os
import pandas as pd

from torch.autograd import Variable
from math import exp
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torchvision.utils import save_image


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1 / torch.sqrt(mse))


target_path = 'A:/temp/temp'
img_path = 'A:/temp/temp'
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    img1_list = os.listdir(target_path)
    img2_list = os.listdir(img_path)

    assert len(img1_list) == len(img2_list), 'the number of images should be same'

    img_zip = zip(img1_list, img2_list)
    ssim = SSIM().cuda()
    psnr = PSNR()

    ssim_list = []
    psnr_list = []

    for idx, items in enumerate(img_zip):
        img1_full_path = os.path.join(target_path, items[0])
        img2_full_path = os.path.join(img_path, items[1])

        img1 = Image.open(img1_full_path).convert('RGB')
        img2 = Image.open(img2_full_path).convert('RGB')

        if 'GAN' in img_path:
            gt_width, gt_height = img1.size
            transform = transforms.Compose([
                transforms.Resize([gt_height, gt_width]),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.ToTensor()

        img1 = transform(img1).unsqueeze(0).cuda()
        img2 = transform(img2).unsqueeze(0).cuda()

        ssim_value = ssim(img1, img2)
        psnr_value = psnr(img1, img2)

        ssim_list.append(ssim_value.cpu().detach().item())
        psnr_list.append(psnr_value.cpu().detach().item())
        print(ssim_value, psnr_value)

    df = {'file_name': img1_list,
          'SSIM': ssim_list,
          'PSNR': psnr_list}

    if 'A' in target_path[-2:]:
        pd.DataFrame(df).to_csv('experiment_evaluation_input.csv', index=False)
    elif 'B' in target_path[-2:]:
        pd.DataFrame(df).to_csv('experiment_evaluation_target.csv', index=False)
    else:
        print('Unrecognized path format')
        pd.DataFrame(df).to_csv('experiment_evaluation.csv', index=False)


if __name__ == "__main__":
    main()
