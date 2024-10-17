import torch
import network
import cv2 as cv
from torchvision.transforms import transforms,ToPILImage,ToTensor
from PIL import Image,ImageOps
import torchvision
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.nn import functional as F
import math
from utils import guide_filter

result_path = r"C:\Users\Administrator\Desktop\code\StepFuse\SwinFusion\MFI-WHU-rgb"
gt_path = r"D:\pythonproject\StepFuse\testdata\MFI-WHU\full_clear"
save_path = r"D:\pythonproject\StepFuse\fusedata\res_map"
def get_imgs_path(path):
    imglist1 = []
    filenames1 = os.listdir(path)
    filenames1.sort(key=lambda x: int(x[:-4]))
    for name in filenames1:
        img_path = path + "/" + name
        imglist1.append(img_path)
    return imglist1


result_imgs_path = get_imgs_path(result_path)
gt_imgs_path = get_imgs_path(gt_path)
index = 0

for a, b in zip(result_imgs_path, gt_imgs_path):
    result_img = Image.open(a).convert("L")
    w,h = result_img.size
    gt_img = Image.open(b).convert("L")
    result_img = ToTensor()(result_img)
    gt_img = ToTensor()(gt_img)
    res_map = gt_img-result_img
    res_map = (res_map - res_map.min()) / (res_map.max() - res_map.min())
    res_map = ToPILImage()(res_map)
    res_img_new = Image.new('RGB', (w, h))
    nr, ng, nb = res_img_new.split()
    res_map = [nr, ng, res_map]
    res_img_new = Image.merge('RGB', res_map)
    final_save_path = os.path.join(save_path + "/" +'%d.png' % (index + 1))
    res_img_new.save(final_save_path)
    index+=1
    print(index)