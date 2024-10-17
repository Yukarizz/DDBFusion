# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from glob import glob
import os
from PIL import Image
import numpy as np
from torchvision import models
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import cv2
_tensor = transforms.ToTensor()
_pil_rgb = transforms.ToPILImage('RGB')
_pil_gray = transforms.ToPILImage()
device = 'cuda'
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])

    return gauss / gauss.sum()
def create_window(window_size, channel,window_type = "gaussion"):
    if window_type=="gaussion":
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    if window_type=="average":
        _2D_window = torch.ones(size=(1,1,window_size,window_size))

    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window
def guide_filter(I,p,window_size,eps=0.1):
    eps = eps**2
    channel = I.shape[1]
    window = create_window(window_size=window_size,channel=I.shape[1],window_type="average")
    if I.is_cuda:
        window = window.cuda(I.get_device())
    pad = nn.ReflectionPad2d(window_size//2)
    mu_i = (F.conv2d(pad(I), window, padding=0, groups=channel)/(window_size*window_size))
    mu_p = (F.conv2d(pad(p), window, padding=0, groups=channel)/(window_size*window_size))
    mu_ip = (F.conv2d(pad(I*p), window, padding=0, groups=channel)/(window_size*window_size))
    cov_ip = mu_ip-mu_i*mu_p
    mu_ii = (F.conv2d(pad(I*I), window, padding=0, groups=channel)/(window_size*window_size))
    var_i = mu_ii-mu_i*mu_i
    a = (cov_ip/(var_i+eps))
    b = mu_p-a*mu_i
    mu_a = (F.conv2d(pad(a), window, padding=0, groups=channel)/(window_size*window_size))
    mu_b = (F.conv2d(pad(b), window, padding=0, groups=channel)/(window_size*window_size))
    GIF_V = mu_a*I+mu_b
    return GIF_V
class Dataset(Data.Dataset):
    def __init__(self, root, resize=256, gray=True):
        self.files = glob(os.path.join(root, '*.*'))
        self.resize = resize
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transforms.ToTensor()
        self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # img = Image.open(self.files[index])
        # if self.gray:
        #     img = img.convert('L')
        # img = self.transform(img)
        img = cv2.imread(self.files[index], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,self.resize)
        (success, saliency_map_ir) = self.saliency.computeSaliency(img)
        img = self.transform(img)
        saliency_map_ir = self.transform(saliency_map_ir)
        return img,saliency_map_ir

class Dataset_TNO(Data.Dataset):
    def __init__(self, root, resize=256, gray=True):
        self.files = glob(os.path.join(root, '*.*'))
        self.resize = resize
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.Resize(size=[resize,resize]),
            transforms.ToTensor()
        ])
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        self.files.sort()
        img = Image.open(self.files[index])
        if self.gray:
            img = img.convert('L')
        img = self.transform(img)
        return img

def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def load_img(img_path, img_type='gray'):
    img = Image.open(img_path)
    if img_type=='gray':
        img = img.convert('L')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)
    return img
# 用拉普拉斯算子进行边缘检测，可用作梯度损失grd_loss = MSE_fun(gradient(img), gradient(img_re))

def nolinear_trans_patch(x):
    # x:[B x Num_patch x patch_size]
    b = torch.rand(1)
    x = torch.log(1+x)*b.item()
    return x


def gradient(input,blur = True):
    conv_op = nn.Conv2d(1, 1, 3, bias=False,padding=1)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # kernel = np.array([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], dtype='float32')
    # kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype='float32')
    kernel = kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = (torch.from_numpy(kernel)).to(device).type(torch.float32)
    gaussion = GaussianBlurConv(channels=1)
    if blur:
        edge_detect = conv_op(gaussion(input))
    else:
        edge_detect = conv_op(input)
    return edge_detect
def gradient_sobel(input):
    conv_op = nn.Conv2d(1, 1, 3, bias=False,padding=1)
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    kernel_T = kernel.T
    kernel = kernel.reshape((1, 1, 3, 3))
    kernel_T = kernel_T.reshape((1, 1, 3, 3))
    conv_op.weight.data = (torch.from_numpy(kernel)).to(device).type(torch.float32)
    edge_detect_vertical = conv_op(input)
    conv_op.weight.data = (torch.from_numpy(kernel_T)).to(device).type(torch.float32)
    edge_detect_horizontal = conv_op(input)
    edge_detect = (edge_detect_vertical+edge_detect_horizontal)
    return edge_detect
def gradient_Isotropic(input):
    conv_op = nn.Conv2d(1, 1, 3, bias=False,padding=1)
    kernel = np.array([[1, 0, -1], [2**0.5, 0, -2**0.5], [1, 0, -1]], dtype='float32')
    kernel_T = np.array([[-1, -2**0.5, -1], [0, 0, 0], [1, 2**0.5, 1]], dtype='float32')
    kernel = kernel.reshape((1, 1, 3, 3))
    kernel_T = kernel_T.reshape((1, 1, 3, 3))
    conv_op.weight.data = (torch.from_numpy(kernel)).to(device).type(torch.float32)
    edge_detect_vertical = conv_op(input)
    conv_op.weight.data = (torch.from_numpy(kernel_T)).to(device).type(torch.float32)
    edge_detect_horizontal = conv_op(input)
    edge_detect = (edge_detect_vertical+edge_detect_horizontal)
    return edge_detect


def hist_similar(x,y):
    t_min = torch.min(torch.cat((x, y), 1)).item()
    t_max = torch.max(torch.cat((x, y), 1)).item()
    return (torch.norm((torch.histc(x, 255, min=t_min, max=t_max)-torch.histc(y, 255, min=t_min, max=t_max)),2))/255


def fusion_exp( a, b):
    expa = torch.exp(a)
    expb = torch.exp(b)
    pa = expa / (expa + expb)
    pb = expb / (expa + expb)

    return pa * a + pb * b







class VGGLoss(nn.Module):
    def __init__(self,end=5):
        super(VGGLoss, self).__init__()
        # device = 'cuda:3'
        self.vgg = Vgg19(end=end).to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0][:end]
        self.e = end

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.151, 0.131, 0.120]).reshape((1, 3, 1, 1))),
                                       requires_grad=False).to(device)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.037, 0.034, 0.031]).reshape((1, 3, 1, 1))),
                                      requires_grad=False).to(device)

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, end = 5,requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.e = end

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out[:self.e]


def mulGANloss(input_, is_real):
    criterionGAN = torch.nn.MSELoss()

    if is_real:
        label = 1
    else:
        label = 0
    # 判断输入是否为list类型
    if isinstance(input_[0], list):
        loss = 0.0
        for i in input_:
            pred = i[-1]
            target = torch.Tensor(pred.size()).fill_(label).to(pred.device)
            loss += criterionGAN(pred, target)
        return loss
    else:
        target = torch.Tensor(input_[-1].size()).fill_(label).to(input_[-1].device)
        return criterionGAN(input_[-1], target)


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device).type(torch.float32)
        self.conv_op = nn.Conv2d(channels,channels,kernel_size=5,stride=1,padding=2,bias=False)
        self.conv_op.weight.data = self.weight
    def forward(self, x):

        return self.conv_op(x)

class Gradient_L1(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(Gradient_L1, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)
