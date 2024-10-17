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
import seaborn as sns
import matplotlib.pyplot as plt
EPSILON = 1e-10

def topatch(img):
    w, h = img.size
    # 在图像的最右边，最下面添加128，直到是224的倍数
    tensor_img = transforms.ToTensor()(img)
    pad = torch.nn.ReflectionPad2d([0,224 - w % 224,0,224 - h % 224])
    tensor_img = pad(tensor_img)
    img_pad = transforms.ToPILImage()(tensor_img)
    # img_pad = ImageOps.expand(img, (0, 0, 224 - w % 224, 224 - h % 224), fill=0)
    nh = (h // 224 + 1)
    nw = (w // 224 + 1)

    cis = []
    for j in range(nh):
        for i in range(nw):
            area = (224 * i, 224 * j, 224 * (i + 1), 224 * (j + 1))
            cropped_img = img_pad.crop(area)
            cis.append(cropped_img)
    return cis

def get_img_path(path1, path2,path3 =None):
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
    if path3 is not None:
        imglist3 = []
        filenames3 = os.listdir(path3)
        filenames3.sort(key=lambda x: int(x[:-4]))
        for name in filenames3:
            img_path = path3 + "/" + name
            imglist3.append(img_path)
        return imglist1, imglist2, imglist3
    return imglist1, imglist2


def saveimg_fuse(img_fuse,class_path, num=0):
    # img = torchvision.utils.make_grid([img1[0].cpu(), img2[0].cpu(), img_fuse[0].cpu()], nrow=3)
    save_path = os.path.join('fusedata/'+class_path+'/fuse1/'+'%d.png' % (num + 1))
    img_fuse.save(save_path)


def saveimg_recon(img, img_re, num=0):
    img = torchvision.utils.make_grid([img[0].cpu(), img_re[0].cpu()], nrow=2)
    torchvision.utils.save_image(img, fp=(os.path.join('fusedata/result_%d.jpg' % (num))))


def loadimg(path1, path2):
    img_vi = Image.open(path1).convert('L')
    img_ir = Image.open(path2).convert('L')
    img_test_vi_cis = topatch(img_vi)
    img_test_ir_cis = topatch(img_ir)
    w, h = img_vi.size
    nh = (h // 224 + 1)
    nw = (w // 224 + 1)
    return img_test_vi_cis,img_test_ir_cis,nh,nw,h,w



def Cross(logits, target):
    logits = torch.sigmoid(logits)
    loss = - target * torch.log(logits) - (1 - target) * torch.log(1 - logits)
    loss = loss.mean()
    return loss.item()


def spatial_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)
    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2
    return tensor_f


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial


def my_softmax(x, y):
    return torch.exp(x) / (torch.exp(x) + torch.exp(y) + EPSILON)
def get_union_weight(x,y):
    return x / (x+y)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda'
    # class_path = "Exposure"
    Class = ['Tno','Road','Lytro',]
    # Class = ['Med','Exposure']
    # Class = ['MFI-WHU']
    fuse_model = network.__dict__['mae_vit_large_patch16'](norm_pix_loss=False)
    fuse_model.mode = "fuse"
    state_dict = torch.load('weights/best_fusion.pt')
    fuse_model.load_state_dict(state_dict['weight'])
    fuse_model.to(device)
    fuse_model.eval()
    for class_path in Class:
        print("model epoch:%d  "%(state_dict['epoch'])+"Processing Dataclass:%s"%class_path)
        entropy = 0
        if class_path == "Exposure" or class_path== "Med":
            path1 = "testdata/" + class_path + "/" + "1"
            path2 = "testdata/" + class_path + "/" + "2"
        elif class_path=="Tno" or class_path == "Road" or class_path=="MFNet":
            path1 = "testdata/" + class_path + "/" + "vi"
            path2 = "testdata/" + class_path + "/" + "ir"
        elif class_path=="Lytro" :
            path1 = "testdata/" + class_path + "/" + "near"
            path2 = "testdata/" + class_path + "/" + "far"
        elif class_path == "MFI-WHU":
            path1 = "testdata/" + class_path + "/" + "near"
            path2 = "testdata/" + class_path + "/" + "far"
            path3 = "testdata/" + class_path + "/" + "full_clear"
            a,b,c = get_img_path(path1,path2,path3)

        n = 0
        a, b = get_img_path(path1, path2)
        # img1是可见光图像，img2是红外图像
        with torch.no_grad():
            for img_path1, img_path2 in zip(a, b):
                img_test_vi_cis,img_test_ir_cis,nh,nw,h,w = loadimg(img_path1, img_path2)
                img_re_cis=[]
                index_cis=0
                for pic_i in range(nw*nh):
                    img_test_vi = transforms.ToTensor()(img_test_vi_cis[pic_i]).to(device)
                    img_test_ir = transforms.ToTensor()(img_test_ir_cis[pic_i]).to(device)

                    img_test_vi = img_test_vi.unsqueeze(0)
                    img_test_ir = img_test_ir.unsqueeze(0)



                    x, y = fuse_model.forward_encoder(img_test_vi, img_test_ir)
                    #[c_in_x,c_in_y,common,trans_c_in_x,trans_c_in_y,xy],gt_pre,gt
                    c_in_x, c_in_y, common, positive_x, nagetive_x, positive_y, nagetive_y, pred = fuse_model.forward_decoder(x,y)  # [N, L, p*p*3]

                    # pred = fuse_model.unpatchify(x)*img_test_vi + fuse_model.unpatchify(y)*img_test_ir + fuse_model.unpatchify(common)
                    pred = fuse_model.unpatchify(pred)
                    # pred = (pred-pred.min())/(pred.max()-pred.min())
                    pred = torch.clamp(pred,min=0,max=1)

                    entropy1 = psnr(img_test_vi.cpu().numpy(), pred.cpu().numpy())
                    entropy2 = psnr(img_test_ir.cpu().numpy(), pred.cpu().numpy())
                    entropy += ((entropy1 + entropy2) / 2)

                    pred = pred.cpu().clone()
                    pred = pred.squeeze(0).squeeze(0)
                    pred = ToPILImage()(pred)
                    img_re_cis.append(pred)
                    result_img = Image.new('L', (w, h))
                for j in range(nh):
                    for i in range(nw):
                        result_img.paste(img_re_cis[index_cis], (224 * i, 224 * j))
                        index_cis+=1

                result_img = ToTensor()(result_img)
                result_img = (result_img-result_img.min())/(result_img.max()-result_img.min())
                '''save heatmap'''
                # result_img = result_img.cpu().squeeze(0).numpy()
                # sns.set_theme()
                # plt.figure(figsize=(w/71, h/71))
                # ax = sns.heatmap(result_img,cbar=False,cmap="rainbow")
                # s1 = ax.get_figure()
                # plt.axes().get_xaxis().set_visible(False)
                # plt.axes().get_yaxis().set_visible(False)
                # s1.savefig("fusedata/heatmap/%s_%s.jpg" % (class_path, str(n)),bbox_inches="tight", pad_inches = 0)
                # plt.close()
                '''save heatmap'''
                '''save fuse image'''
                result_img = ToPILImage()(result_img)
                saveimg_fuse(img_fuse=result_img,class_path=class_path, num=n)
                '''save fuse image'''
                n += 1
        print("Down! Total Psnr:%.5f" % float(entropy/n))




