# import math
import torch.nn as nn
# from torch.nn.modules.utils import _triple
import torch
import torch.nn.functional as F
from itertools import repeat

import collections.abc as container_abcs

class DownBlock(nn.Module):
    def __init__(self, num_convs, inchannels, outchannels,dim,pool=True):
        super(DownBlock, self).__init__()
        blk = []
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for i in range(num_convs):
            if i == 0:
                blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
                blk.append(nn.BatchNorm2d(outchannels))
            else:
                blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
                blk.append(nn.BatchNorm2d(outchannels))
            blk.append(nn.GELU())
        self.layer = nn.Sequential(*blk)

    def forward(self, x):
        return self.layer(x)

class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels,dim,skip=True):
        super(UpBlock, self).__init__()
        self.convt = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.GELU(),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.convt(x)
        x = self.conv(x)
        return x

def conv_norm_lrelu(input_dim, output_dim,mode='conv', kernel_size=(3,3),padding=(1,1), stride=(1,1),bias=False,normtype='bn'):
    if mode=='down':
        conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=(2,2), padding=padding, bias=bias)
    elif mode == 'up':
        conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=(2,2), padding=padding, bias=bias)
    else:  # 如果不是上采样或者下采样，正常选用这个if分支，即普通的卷积
        conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    if normtype == 'bn':
        norm = nn.BatchNorm2d(output_dim)
    else:
        norm = nn.InstanceNorm2d(output_dim)
    lrelu = nn.LeakyReLU()
    layer = [conv, norm, lrelu]
    return nn.Sequential(*layer)

def res_conv_norm_lrelu(input_dim, output_dim,mode='conv', kernel_size=(3,3),padding=(1,1), stride=(1,1),bias=False,normtype='bn'):
    if mode=='down':
        conv = nn.Conv2d(input_dim, output_dim, kernel_size=(2,2), stride=(2,2), padding=(0,0), bias=bias)
    elif mode == 'up':
        conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
    else:
        conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    if normtype == 'bn':
        norm = nn.BatchNorm2d(output_dim)
    else:
        norm = nn.InstanceNorm2d(output_dim)
    lrelu = nn.LeakyReLU()
    # layer = [conv, norm, lrelu,ResBlock(output_dim),ResBlock(output_dim),ResBlock(output_dim)]
    layer = [conv,norm,lrelu,ResBlock(output_dim)]
    return nn.Sequential(*layer)



def conv1x1(input_dim,output_dim):
    conv = nn.Conv2d(input_dim,output_dim,kernel_size=(1,1))
    return conv


class ResBlock(nn.Module):
    def __init__(self, dim, norm="batch", activation="lrelu", pad_type="reflect"):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
    ):
        super().__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == "batch":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "instance":
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class SSEBlock(nn.Module):
    def __init__(self,feature_dim,output_dim):
        super(SSEBlock, self).__init__()
        self.BN = nn.BatchNorm2d(feature_dim)
        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv1 = conv1x1(feature_dim, output_dim)

        self.activation = nn.Sigmoid()

    # input is 256*16*16
    def forward(self, x):
        x = self.BN(x)
        x = self.Avgpool(x)

        x = self.conv1(x)

        x = self.activation(x)
        return x


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob  # 每个元素失活的可能性
        self.block_size = block_size  # 失活Block的大小

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output 保证做DropBlock之前和做DropBlock 之后，该层都具体相同的均值，方差。
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            # 如果block大小是2的话,边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)




#  下面这些暂时不用管
# class UpsampleReshape_train(torch.nn.Module):
#     def __init__(self,inp_c,out_c,is_fuse=False):
#         super(UpsampleReshape_train, self).__init__()
#         if is_fuse:
#             self.up = conv_norm_lrelu(inp_c, out_c,kernel_size=4,mode='up' )
#         else:
#             self.up = nn.ConvTranspose2d(inp_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
#
#     def forward(self, y,x):
#         return self.up(x)
#
# class UpsampleReshape_eval(torch.nn.Module):
#     def __init__(self,inp_c,out_c,is_fuse=False):
#         super(UpsampleReshape_eval, self).__init__()
#         if is_fuse:
#             self.up = conv_norm_lrelu(inp_c, out_c, kernel_size=4, mode='up')
#         else:
#             self.up = nn.ConvTranspose2d(inp_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
#
#     def forward(self, x1,x2):
#         x2 = self.up(x2)
#         # print(x2.shape,x1.shape,"x")
#         shape_x1 = x1.size()
#         shape_x2 = x2.size()
#         left = 0
#         right = 0
#         top = 0
#         bot = 0
#         if shape_x1[3] != shape_x2[3]:
#             lef_right = shape_x1[3] - shape_x2[3]
#             if lef_right%2 is 0.0:
#                 left = int(lef_right/2)
#                 right = int(lef_right/2)
#             else:
#                 left = int(lef_right / 2)
#                 right = int(lef_right - left)
#
#         if shape_x1[2] != shape_x2[2]:
#             top_bot = shape_x1[2] - shape_x2[2]
#             if top_bot%2 is 0.0:
#                 top = int(top_bot/2)
#                 bot = int(top_bot/2)
#             else:
#                 top = int(top_bot / 2)
#                 bot = int(top_bot - top)
#
#         reflection_padding = [left, right, top, bot]
#         reflection_pad = nn.ReflectionPad2d(reflection_padding)
#         x2 = reflection_pad(x2)
#         # print(x2.shape, x1.shape, "x2")
#         return x2