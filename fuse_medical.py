import torch
import network
from tqdm import tqdm
from tqdm.contrib import tzip
from torchvision.transforms import transforms,ToPILImage,ToTensor
from PIL import Image,ImageOps
import torchvision
import os
import matplotlib
from torch.nn import functional as F
import math
import numpy as np
from utils import guide_filter
import seaborn as sns
from matplotlib import pyplot as plt
EPSILON = 1e-10
img_size = 224
def topatch(img):
    w, h = img.size
    # 在图像的最右边，最下面添加128，直到是224的倍数
    tensor_img = transforms.ToTensor()(img)
    pad = torch.nn.ReflectionPad2d([0, 224 - w % 224, 0, 224 - h % 224])
    tensor_img = pad(tensor_img)
    img_pad = transforms.ToPILImage()(tensor_img)
    # img_pad = ImageOps.expand(img, (0, 0, img_size - w % img_size, img_size - h % img_size), fill=0)
    nh = (h // img_size + 1)
    nw = (w // img_size + 1)

    cis = []
    for j in range(nh):
        for i in range(nw):
            area = (img_size * i, img_size * j, img_size * (i + 1), img_size * (j + 1))
            cropped_img = img_pad.crop(area)
            cis.append(cropped_img)
    return cis

def get_img_path(path1, path2):
    imglist1 = []
    imglist2 = []
    filenames1 = os.listdir(path1)
    filenames2 = os.listdir(path2)
    if "MFNet" in path1:
        filenames1.sort(key=lambda x: int(x[:-5]))
        filenames2.sort(key=lambda x: int(x[:-5]))
    else:
        filenames1.sort(key=lambda x: int(x[:-4]))
        filenames2.sort(key=lambda x: int(x[:-4]))
    for name in filenames1:
        img_path = path1 + "/" + name
        imglist1.append(img_path)
    for name in filenames2:
        img_path = path2 + "/" + name
        imglist2.append(img_path)
    return imglist1, imglist2


def saveimg_fuse(result_img,class_path, num=0):
    # img = torchvision.utils.make_grid([img1[0].cpu(), img2[0].cpu(), img_fuse[0].cpu()], nrow=3)
    save_path = os.path.join('fusedata/'+class_path+'/fuse1/'+'%d.png' % (num + 1))
    result_img.save(save_path)



def loadimg(path1, path2):
    A = [[1. / 3., 1. / 3., 1. / 3.], [-np.sqrt(2) / 6., -np.sqrt(2) / 6., 2 * np.sqrt(2) / 6],
         [1. / np.sqrt(2), -1. / np.sqrt(2), 0.]]
    # RGB－>IHS正变换矩阵
    B = [[1., -1. / np.sqrt(2), 1. / np.sqrt(2)], [1., -1. / np.sqrt(2), -1. / np.sqrt(2)], [1., np.sqrt(2), 0.]]
    # IHS－>RGB逆变换矩阵
    A = np.matrix(A)
    B = np.matrix(B)


    img_vi = np.array(Image.open(path1))
    img_ir = Image.open(path2)
    h, w, band = img_vi.shape
    img_vi = img_vi.reshape((h*w,3))
    a1 = np.dot(A, np.matrix(img_vi).T)
    a1 = np.array(a1).reshape(3,h,w)
    I,H,S = a1[0],a1[1],a1[2]
    I = Image.fromarray(I/255)

    img_test_vi_cis = topatch(I)
    img_test_ir_cis = topatch(img_ir)

    nh = (h // img_size + 1)
    nw = (w // img_size + 1)
    return img_test_vi_cis,img_test_ir_cis,[nh,nw,h,w],[H,S]


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda'
    Class = ['Med']
    fuse_model = network.__dict__['mae_vit_large_patch16'](norm_pix_loss=False)
    fuse_model.mode = "fuse"
    state_dict = torch.load('weights/best_fusion.pt')
    fuse_model.load_state_dict(state_dict['weight'])
    fuse_model.to(device)
    fuse_model.eval()

    total = sum([params.nelement() for params in fuse_model.parameters()])
    print("Number of params: {%.2f M}" % (total / 1e6))

    for class_path in Class:
        print("model epoch:%d  "%(state_dict['epoch'])+"Processing Dataclass:%s"%class_path)
        entropy = 0
        if class_path == "Exposure" or class_path == "Med":
            path1 = "testdata/" + class_path + "/" + "1"
            path2 = "testdata/" + class_path + "/" + "2"
        elif  class_path=="Lytro" or class_path=="MFI-WHU":
            path1 = "testdata/" + class_path + "/" + "near"
            path2 = "testdata/" + class_path + "/" + "far"
        elif class_path=="MFNet":
            path1 = "testdata/" + class_path + "/" + "vi"
            path2 = "testdata/" + class_path + "/" + "ir"
        n = 0
        a, b = get_img_path(path1, path2)

        tqdms = tqdm(tzip(a, b))
        with torch.no_grad():
            for img_path1, img_path2 in tqdms:
                img_test_vi_cis,img_test_ir_cis,shape,HS = loadimg(img_path1, img_path2)
                nh, nw, h, w = shape
                img_re_cis=[]
                tqdms.set_description("%sth img"%(n))
                for pic_i in range(nh*nw):
                    img_test_vi = transforms.ToTensor()(img_test_vi_cis[pic_i]).to(device)
                    img_test_ir = transforms.ToTensor()(img_test_ir_cis[pic_i]).to(device)
                    img_test_vi = img_test_vi.unsqueeze(0)
                    img_test_ir = img_test_ir.unsqueeze(0)
                    vi_b = guide_filter(I=img_test_vi, p=img_test_vi, window_size=11, eps=0.2)
                    ir_b = guide_filter(I=img_test_ir, p=img_test_ir, window_size=11, eps=0.2)


                    x, y = fuse_model.forward_encoder(img_test_vi, img_test_ir)
                    c_in_x, c_in_y, common, positive_x, nagetive_x, positive_y, nagetive_y, pred = fuse_model.forward_decoder(x,y)

                    pred = fuse_model.unpatchify(pred)


                    pred = torch.clamp(pred, min=0, max=1)
                    # || end ||
                    pred = pred.cpu().clone()
                    pred = pred.squeeze(0).squeeze(0)
                    pred = ToPILImage()(pred)
                    img_re_cis.append(pred)
                result_img = Image.new('L', (w, h))

                index_cis = 0
                for j in range(nh):
                    for i in range(nw):
                        result_img.paste(img_re_cis[index_cis], (img_size * i, img_size * j))
                        index_cis = index_cis+1
                result_img = ToTensor()(result_img)
                result_img = (result_img-result_img.min())/(result_img.max()-result_img.min())
                result_img = ToPILImage()(result_img)
                '''save heatmap'''
                # result_img = result_img.cpu().squeeze(0).numpy()
                # sns.set_theme()
                # plt.figure(figsize=(nw, nh))
                # ax = sns.heatmap(result_img, cbar=False, cmap="rainbow")
                # s1 = ax.get_figure()
                # plt.axes().get_xaxis().set_visible(False)
                # plt.axes().get_yaxis().set_visible(False)
                # s1.savefig("fusedata/heatmap/%s_%s.jpg" % (class_path, str(n)), bbox_inches="tight", pad_inches=0)
                # plt.close()
                '''save heatmap'''

                saveimg_fuse(result_img, class_path=class_path, num=n)
                n = n+1



