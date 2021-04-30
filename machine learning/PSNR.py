import torch
import os
import pandas as pd

from PIL import Image
from torchvision import transforms


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1 / torch.sqrt(mse))


target_path = 'A:/Users/SSY/Desktop/dataset/cud_calibration/210308_paper_dataset/dataset/test/A'
img_path = 'A:/Users/SSY/Desktop/Other works/paper/졸업논문/experiments/CUD-Net/model14_iden'


def main():
    img1_list = os.listdir(target_path)
    img2_list = os.listdir(img_path)

    assert len(img1_list) == len(img2_list), 'the number of images should be same'

    img_zip = zip(img1_list, img2_list)
    psnr = PSNR()

    psnr_list = []

    for idx, items in enumerate(img_zip):
        img1_full_path = os.path.join(target_path, items[0])
        img2_full_path = os.path.join(img_path, items[1])

        img1 = Image.open(img1_full_path).convert('RGB')
        img2 = Image.open(img2_full_path).convert('RGB')

        transform = transforms.ToTensor()

        img1 = transform(img1).unsqueeze(0)
        img2 = transform(img2).unsqueeze(0)

        psnr_value = psnr(img1, img2)

        psnr_list.append(psnr_value.cpu().detach().item())
        print(psnr_value)

    df = {'file_name': img1_list,
          'A_percentage': psnr_list}

    pd.DataFrame(df).to_csv('psnr_result.csv', index=False)


if __name__ == "__main__":
    main()
