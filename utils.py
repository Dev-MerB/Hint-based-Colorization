import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import natsort
import cv2
import os
from tqdm import tqdm

def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0))), 0, 1) * 255.0
    return image_numpy.astype(imtype)

def pil2tensor(im):  # in: [PIL Image with 3 channels]. out: [B=1, C=3, H, W] (0, 1)
    return torch.Tensor((np.float32(im) / 255).transpose(2, 0 ,1)).unsqueeze(0)

def PSNR_SSIM(GT_path, Pred_Path):
    GT_list = natsort.natsorted(os.listdir(GT_path))
    Pred_list = natsort.natsorted(os.listdir(Pred_Path))

    psnr, ssim = [], []
    for GT, Pred in tqdm(zip(GT_list,Pred_list),total=len(GT_list)):
        GT = cv2.imread(os.path.join(GT_path,GT))
        Pred =cv2.imread(os.path.join(Pred_Path,Pred))

        psnr.append(peak_signal_noise_ratio(GT,Pred))
        ssim.append(structural_similarity(GT,Pred, channel_axis=2))
    print("PSNR : {} SSIM: {}".format(np.average(psnr),np.average(ssim)))