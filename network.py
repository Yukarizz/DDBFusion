# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import random
from functools import partial
from torchvision.utils import save_image
import torch
import torch.nn as nn
from vit_utils import PatchEmbed,Block,Mlp
from einops import rearrange
from pos_embed import get_2d_sincos_pos_embed
from utils import gradient,nolinear_trans_patch,create_window
from swin_utils import SwinTransformerBlock
import numpy as np
import torch.nn.functional as F
import math
from math import factorial
from scipy.interpolate import interp1d

def no_linear_trans(x):
    return 0.5 * (torch.sin(math.pi * x + math.pi / 2) + 1)


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,trans_patch_size=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mode = "train"):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        self.trans_patch_size=[1, 2, 4, 7, 8, 14, 16, 28, 32]
        self.patch_embed_one = PatchEmbed(img_size, 1, in_chans, embed_dim)
        self.patch_embed_two = PatchEmbed(img_size, 1, in_chans, embed_dim)
        self.num_patches = self.patch_embed_one.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim),
                                      requires_grad=True)

        # self.Encoder_u_one = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(depth // 2)])
        # self.Encoder_u_two = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(depth // 2)])


        self.Encoder_u_one = nn.ModuleList([
            SwinTransformerBlock(dim=embed_dim, input_resolution=(img_size,img_size),
                                 num_heads=num_heads, window_size=7,
                                 shift_size=0 if (i % 2 == 0) else 7 // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=True, qk_scale=None,
                                 drop=0., attn_drop=0.,
                                 norm_layer=norm_layer)
            for i in range(depth//2)])
        self.Encoder_u_two = nn.ModuleList([
            SwinTransformerBlock(dim=embed_dim, input_resolution=(img_size, img_size),
                                 num_heads=num_heads, window_size=7,
                                 shift_size=0 if (i % 2 == 0) else 7 // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=True, qk_scale=None,
                                 drop=0., attn_drop=0.,
                                 norm_layer=norm_layer)
            for i in range(depth//2)])



        self.norm_x = norm_layer(embed_dim)
        self.norm_y = norm_layer(embed_dim)
        # self.norm_xy = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics


        self.mode = mode

         # decoder to patch
        self.decoder_embed = Mlp(embed_dim*5, embed_dim * 4, embed_dim)
        # self.decoder_blocks = nn.ModuleList([
        #     SwinTransformerBlock(dim=embed_dim, input_resolution=(img_size, img_size),
        #                          num_heads=num_heads, window_size=7,
        #                          shift_size=0 if (i % 2 == 0) else 7 // 2,
        #                          mlp_ratio=mlp_ratio,
        #                          qkv_bias=True, qk_scale=None,
        #                          drop=0., attn_drop=0.,
        #                          norm_layer=norm_layer)
        #     for i in range(decoder_depth//2)])
        # self.decoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim),
        #                               requires_grad=True)
        # self.decoder_norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans)


        self.c_in_x_embd = Mlp(embed_dim*2,embed_dim*2,embed_dim)
        self.c_in_y_embd = Mlp(embed_dim * 2, embed_dim * 2, embed_dim)
        self.c_in_x_pred = Mlp(embed_dim, embed_dim * 2, patch_size ** 2 * in_chans)
        self.c_in_y_pred = Mlp(embed_dim, embed_dim * 2, patch_size ** 2 * in_chans)

        self.positive_x_embd = Mlp(embed_dim*2,embed_dim*2,embed_dim)
        self.positive_x_pred = Mlp(embed_dim, embed_dim * 2, patch_size ** 2 * in_chans)
        self.nagetive_x_embd = Mlp(embed_dim*2, embed_dim * 2, embed_dim)
        self.nagetive_x_pred = Mlp(embed_dim, embed_dim * 2, patch_size ** 2 * in_chans)

        self.positive_y_embd = Mlp(embed_dim*2, embed_dim * 2, embed_dim)
        self.positive_y_pred = Mlp(embed_dim, embed_dim * 2, patch_size ** 2 * in_chans)
        self.nagetive_y_embd = Mlp(embed_dim*2, embed_dim * 2, embed_dim)
        self.nagetive_y_pred = Mlp(embed_dim, embed_dim * 2, patch_size ** 2 * in_chans)

        self.common_embd = Mlp(embed_dim*2,embed_dim*2,embed_dim)
        self.common = Mlp(embed_dim, embed_dim * 2, patch_size ** 2 * in_chans)


        # self.clear_embd = Mlp(embed_dim*2,embed_dim*2,embed_dim)
        # self.clear = Mlp(embed_dim,embed_dim*2,patch_size ** 2 * in_chans)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
        #                                     cls_token=True)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        #
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.patch_embed.num_patches ** .5), cls_token=True)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w_1 = self.patch_embed_one.proj.weight.data
        torch.nn.init.xavier_uniform_(w_1.view([w_1.shape[0], -1]))
        w_2 = self.patch_embed_two.proj.weight.data
        torch.nn.init.xavier_uniform_(w_2.view([w_1.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))
        return x

    def unpatchify(self, x,channel = 1):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed_one.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, channel))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channel, h * p, h * p))
        return imgs



    def forward_encoder(self, img_one, img_two):
        # embed patches
        x = self.patch_embed_one(img_one)
        y = self.patch_embed_two(img_two)
        # add pos embed w/o cls token
        x = x + self.pos_embed
        y = y + self.pos_embed

        # apply Transformer blocks
        for blk in self.Encoder_u_one:
            x = blk(x)
        x = self.norm_x(x)

        for blk in self.Encoder_u_two:
            y = blk(y)
        y = self.norm_y(y)
        return x,y

    def forward_decoder(self, x,y):
        # x,y : 50176,96
        res = torch.cat([x, y], dim=2)
        common_embd = self.common_embd(res)
        common = self.common(common_embd)

        c_in_x_embd = self.c_in_x_embd(res)
        c_in_x = self.c_in_x_pred(c_in_x_embd)
        c_in_y_embd = self.c_in_y_embd(res)
        c_in_y = self.c_in_y_pred(c_in_y_embd)

        positive_x_embd = self.positive_x_embd(torch.cat([c_in_x_embd,c_in_y_embd],dim=2))
        positive_x = self.positive_x_pred(positive_x_embd)

        positive_y_embd = self.positive_y_embd(torch.cat([c_in_x_embd,c_in_y_embd],dim=2))
        positive_y = self.positive_y_pred(positive_y_embd)

        nagetive_x_embd = self.nagetive_x_embd(torch.cat([c_in_x_embd,c_in_y_embd],dim=2))
        nagetive_x = self.nagetive_x_pred(positive_x_embd)

        nagetive_y_embd = self.nagetive_y_embd(torch.cat([c_in_x_embd,c_in_y_embd],dim=2))
        nagetive_y = self.nagetive_y_pred(positive_y_embd)

        xy = torch.cat([positive_x_embd, positive_y_embd,nagetive_x_embd,nagetive_y_embd, common_embd], dim=2)
        # xy = self.norm_xy(xy)
        # # embed tokens
        xy = self.decoder_embed(xy)
        # # add pos embed
        # xy = xy + self.decoder_pos
        # # # # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     xy = blk(xy)
        # xy = self.decoder_norm(xy)
        # # # predictor projection
        xy = self.decoder_pred(xy)

        return c_in_x,c_in_y,common,positive_x,nagetive_x,positive_y,nagetive_y,xy

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def Bezier_curve(self,points):
        N = len(points)
        n = N - 1
        px = []
        py = []
        for T in range(1001):
            t = T * 0.001
            x, y = 0, 0
            for i in range(N):
                B = factorial(n) * t ** i * (1 - t) ** (n - i) / (factorial(i) * factorial(n - i))
                x += points[i][0] * B
                y += points[i][1] * B
            px.append(x)
            py.append(y)
        f = interp1d(px, py, kind='linear', axis=-1)
        return f

    def random_pix_trans(self,imgs,trans_patch_size):
    #     imgs : [b c h w]
        # N, L, D = imgs.shape  # batch, length, dim
        window = create_window(window_size=11,channel=1,window_type="gaussion")
        if imgs.is_cuda:
            window = window.cuda(imgs.get_device())
        window = window.type_as(imgs)
        pad = nn.ReflectionPad2d(11 // 2)

        blured_img = F.conv2d(pad(imgs), window, padding=0, groups=1)

        blured_img = rearrange(blured_img,"b c (nh bzh) (nw bzw)-> b c (nh nw) (bzh bzw)",bzh=trans_patch_size,bzw=trans_patch_size)
        imgs = rearrange(imgs,"b c (nh bzh) (nw bzw)-> b c (nh nw) (bzh bzw)",bzh=trans_patch_size,bzw=trans_patch_size)

        imgs = imgs.squeeze(dim=1)
        blured_img = blured_img.squeeze(dim=1)
        b,num_patchs,d = imgs.shape

        noise = torch.rand(b, num_patchs, device=imgs.device)

        a = np.random.random(5)
        a /= a.sum()

        # a = [0.1, 0.1, 0.1, 0.6, 0.1]



        len_xy_cb = int(num_patchs*a[0])
        len_xy_bc = int(num_patchs*a[1])
        # len_xy_cc = int(196*a[2])
        len_xy_cc_one = int(num_patchs*a[2])
        len_xy_cc_two = int(num_patchs*a[3])
        len_keep = int(num_patchs*a[4])
        len_trans = len_xy_cb+len_xy_bc+len_xy_cc_one+len_xy_cc_two

        len_noise = int(len_trans/2)
        len_one_noise = int(len_noise/2)
        len_two_noise = len_noise-len_one_noise


        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle,dim=1)
        ''' EW : E需要增强，W需要削弱 对应 A*trans B/trans '''
        ids_xy_cb = ids_shuffle[:, :len_xy_cb]
        ids_xy_bc = ids_shuffle[:, len_xy_cb:len_xy_cb+len_xy_bc]
        ids_xy_cc_one = ids_shuffle[:,len_xy_cb+len_xy_bc:len_xy_cb+len_xy_bc+len_xy_cc_one]
        ids_xy_cc_two = ids_shuffle[:,len_xy_cb+len_xy_bc+len_xy_cc_one:len_xy_cb+len_xy_bc+len_xy_cc_one+len_xy_cc_two]

        ids_keep = ids_shuffle[:,len_xy_cb+len_xy_bc+len_xy_cc_one+len_xy_cc_two:]
        ids_one_noise = ids_shuffle[:,:len_one_noise]
        ids_two_noise = ids_shuffle[:,len_one_noise:len_one_noise+len_two_noise]
        ids_noise_bar = ids_shuffle[:, len_one_noise + len_two_noise:]
        # ids_one_blur = ids_shuffle[:,len_one_noise+len_two_noise:len_one_noise+len_two_noise+len_one_blur]
        # ids_two_blur = ids_shuffle[:,len_one_noise+len_two_noise+len_one_blur:len_one_noise+len_two_noise+len_one_blur+len_two_blur]

        # ||| get ground truth |||
        img_x_cb_c = torch.gather(imgs, dim=1, index=ids_xy_cb.unsqueeze(-1).repeat(1, 1, d))
        img_y_cb_b = torch.gather(blured_img, dim=1, index=ids_xy_cb.unsqueeze(-1).repeat(1, 1, d))
        img_x_bc_b = torch.gather(blured_img, dim=1, index=ids_xy_bc.unsqueeze(-1).repeat(1, 1, d))
        img_y_bc_c = torch.gather(imgs, dim=1, index=ids_xy_bc.unsqueeze(-1).repeat(1, 1, d))
        img_x_cc_one = torch.gather(blured_img, dim=1, index=ids_xy_cc_one.unsqueeze(-1).repeat(1, 1, d))
        img_y_cc_one = torch.gather(imgs, dim=1, index=ids_xy_cc_one.unsqueeze(-1).repeat(1, 1, d))
        img_x_cc_two = torch.gather(imgs, dim=1, index=ids_xy_cc_two.unsqueeze(-1).repeat(1, 1, d))
        img_y_cc_two = torch.gather(blured_img, dim=1, index=ids_xy_cc_two.unsqueeze(-1).repeat(1, 1, d))
        img_keep = torch.gather(imgs, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
        # ||| end |||

        # ||| start trans|||
        # positive curve
        p1 = (0, 0)
        p4 = (1, 1)
        index_x = random.uniform(0, 1)
        p2 = (index_x, random.uniform(index_x, 1))
        index_x = random.uniform(0, 1)
        p3 = (index_x, random.uniform(index_x, 1))
        index_x = random.uniform(0, 1)
        p5 = (index_x, random.uniform(index_x, 1))
        index_x = random.uniform(0, 1)
        p6 = (index_x, random.uniform(index_x, 1))
        # negative curve
        pp1 = (0, 0)
        pp4 = (1, 1)
        index_x = random.uniform(0, 1)
        pp2 = (index_x, random.uniform(0, index_x))
        index_x = random.uniform(0, 1)
        pp3 = (index_x, random.uniform(0, index_x))
        index_x = random.uniform(0, 1)
        pp5 = (index_x, random.uniform(0, index_x))
        index_x = random.uniform(0, 1)
        pp6 = (index_x, random.uniform(0, index_x))
        positive_curve_one = self.Bezier_curve([p1,p2,p3,p4])
        positive_curve_two = self.Bezier_curve([p1,p5,p6,p4])
        negative_curve_one = self.Bezier_curve([pp1,pp2,pp3,pp4])
        negative_curve_two = self.Bezier_curve([pp1, pp5, pp6, pp4])

        img_x_cb_c_T = torch.tensor(positive_curve_one(img_x_cb_c.cpu()),dtype=torch.float32).to(imgs.device)
        img_y_cb_b_T = torch.tensor(negative_curve_one(img_y_cb_b.cpu()),dtype=torch.float32).to(imgs.device)
        img_x_bc_b_T = torch.tensor(negative_curve_one(img_x_bc_b.cpu()),dtype=torch.float32).to(imgs.device)
        img_y_bc_c_T = torch.tensor(positive_curve_one(img_y_bc_c.cpu()),dtype=torch.float32).to(imgs.device)

        img_x_cc_one_T = torch.tensor(positive_curve_two(img_x_cc_one.cpu()),dtype=torch.float32).to(imgs.device)
        img_y_cc_one_T = torch.tensor(negative_curve_two(img_y_cc_one.cpu()),dtype=torch.float32).to(imgs.device)
        img_x_cc_two_T = torch.tensor(negative_curve_two(img_x_cc_two.cpu()),dtype=torch.float32).to(imgs.device)
        img_y_cc_two_T = torch.tensor(positive_curve_two(img_y_cc_two.cpu()),dtype=torch.float32).to(imgs.device)

        img_x = torch.cat([img_x_cb_c_T,img_x_bc_b_T,img_x_cc_one_T,img_x_cc_two_T,img_keep],dim=1)
        img_y = torch.cat([img_y_cb_b_T,img_y_bc_c_T,img_y_cc_one_T,img_y_cc_two_T,img_keep],dim=1)

        # ||| end trans |||

        # ||| get ground truth |||
        zeros_like_cb = torch.zeros_like(img_x_cb_c, device=imgs.device)
        zeros_like_bc = torch.zeros_like(img_x_bc_b, device=imgs.device)
        zeros_like_cc_one = torch.zeros_like(img_x_cc_one, device=imgs.device)
        zeros_like_cc_two = torch.zeros_like(img_x_cc_two, device=imgs.device)
        zeros_like_keep = torch.zeros_like(img_keep, device=imgs.device)

        c_in_x = torch.cat([img_x_cb_c_T,zeros_like_bc,zeros_like_cc_one,img_x_cc_two_T,zeros_like_keep],dim=1)
        c_in_y = torch.cat([zeros_like_cb,img_y_bc_c_T,img_y_cc_one_T,zeros_like_cc_two,zeros_like_keep],dim=1)

        gt_x_positive = torch.cat([img_x_cb_c_T,zeros_like_bc,zeros_like_cc_one,zeros_like_cc_two,zeros_like_keep],dim=1)
        gt_x_negative = torch.cat([zeros_like_cb, zeros_like_bc, zeros_like_cc_one, img_x_cc_two_T, zeros_like_keep], dim=1)

        gt_y_positive = torch.cat([zeros_like_cb, img_y_bc_c_T, zeros_like_cc_one,zeros_like_cc_two, zeros_like_keep], dim=1)
        gt_y_negative = torch.cat([zeros_like_cb, zeros_like_bc, img_y_cc_one_T, zeros_like_cc_two, zeros_like_keep], dim=1)
        gt_common = torch.cat([zeros_like_cb, zeros_like_bc, zeros_like_cc_one,zeros_like_cc_two, img_keep], dim=1)
        gt_final = torch.cat([img_x_cb_c_T,img_y_bc_c_T,img_y_cc_one_T,img_x_cc_two_T,img_keep],dim=1)

        img_x = torch.gather(img_x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))
        img_y = torch.gather(img_y, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))
        noise_one_pos = torch.gather(img_x, dim=1, index=ids_one_noise.unsqueeze(-1).repeat(1, 1, d))
        noise_two_neg = torch.gather(img_y, dim=1, index=ids_one_noise.unsqueeze(-1).repeat(1, 1, d))
        noise_one_neg = torch.gather(img_x, dim=1, index=ids_two_noise.unsqueeze(-1).repeat(1, 1, d))
        noise_two_pos = torch.gather(img_y, dim=1, index=ids_two_noise.unsqueeze(-1).repeat(1, 1, d))
        res_one_part = torch.gather(img_x, dim=1, index=ids_noise_bar.unsqueeze(-1).repeat(1, 1, d))
        res_two_part = torch.gather(img_y, dim=1, index=ids_noise_bar.unsqueeze(-1).repeat(1, 1, d))


        noise_one_pos = noise_one_pos + torch.randint_like(noise_one_pos,low=0,high=100)/100*1e-2
        noise_two_neg = noise_two_neg - torch.randint_like(noise_two_neg,low=0,high=100)/100*1e-2
        noise_one_neg = noise_one_neg - torch.randint_like(noise_one_neg,low=0,high=100)/100*1e-2
        noise_two_pos = noise_two_pos + torch.randint_like(noise_two_pos,low=0,high=100)/100*1e-2
        img_x = torch.cat([noise_one_pos,noise_one_neg,res_one_part],dim=1)
        img_y = torch.cat([noise_two_neg,noise_two_pos,res_two_part],dim=1)

        img_x = torch.gather(img_x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))
        img_y = torch.gather(img_y, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))
        c_in_x = torch.gather(c_in_x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))
        c_in_y = torch.gather(c_in_y, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))
        gt_x_positive = torch.gather(gt_x_positive, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))
        gt_x_negative = torch.gather(gt_x_negative, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))
        gt_y_positive = torch.gather(gt_y_positive, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))
        gt_y_negative = torch.gather(gt_y_negative, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))
        gt_common = torch.gather(gt_common, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))
        gt_final = torch.gather(gt_final, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, imgs.shape[2]))

        img_x = rearrange(img_x, "b (nh nw) (p1 p2)->b (nh p1) (nw p2)", p1=trans_patch_size, nh=int(224/trans_patch_size)).unsqueeze(1)
        img_y = rearrange(img_y, "b (nh nw) (p1 p2)->b (nh p1) (nw p2)", p1=trans_patch_size, nh=int(224/trans_patch_size)).unsqueeze(1)
        c_in_x = rearrange(c_in_x, "b (nh nw) (p1 p2)->b (nh p1) (nw p2)", p1=trans_patch_size, nh=int(224/trans_patch_size)).unsqueeze(1)
        c_in_y = rearrange(c_in_y, "b (nh nw) (p1 p2)->b (nh p1) (nw p2)", p1=trans_patch_size, nh=int(224/trans_patch_size)).unsqueeze(1)
        gt_x_positive = rearrange(gt_x_positive, "b (nh nw) (p1 p2)->b (nh p1) (nw p2)", p1=trans_patch_size, nh=int(224/trans_patch_size)).unsqueeze(1)
        gt_x_negative = rearrange(gt_x_negative, "b (nh nw) (p1 p2)->b (nh p1) (nw p2)", p1=trans_patch_size, nh=int(224/trans_patch_size)).unsqueeze(1)
        gt_y_positive = rearrange(gt_y_positive, "b (nh nw) (p1 p2)->b (nh p1) (nw p2)", p1=trans_patch_size, nh=int(224/trans_patch_size)).unsqueeze(1)
        gt_y_negative = rearrange(gt_y_negative, "b (nh nw) (p1 p2)->b (nh p1) (nw p2)", p1=trans_patch_size, nh=int(224/trans_patch_size)).unsqueeze(1)
        gt_common = rearrange(gt_common, "b (nh nw) (p1 p2)->b (nh p1) (nw p2)", p1=trans_patch_size, nh=int(224/trans_patch_size)).unsqueeze(1)
        gt_final = rearrange(gt_final, "b (nh nw) (p1 p2)->b (nh p1) (nw p2)", p1=trans_patch_size, nh=int(224/trans_patch_size)).unsqueeze(1)
        # img_one = torch.clamp(img_one,min=0,max=1)
        # img_two = torch.clamp(img_two, min=0, max=1)

        # save_image(img_x,'img_one.png')
        # save_image(img_y, 'img_two.png')

        # save_image(c_in_x,'c_inx.png')
        # save_image(c_in_y,'c_iny.png')
        # save_image(gt_common,'gt_c.png')
        # save_image(gt_x_positive+gt_x_negative+gt_y_positive+gt_y_negative+gt_common,'img.png')
        return [img_x,img_y],[c_in_x,c_in_y],[gt_x_positive,gt_x_negative,gt_y_positive,gt_y_negative,gt_common,gt_final]

    def forward(self, imgs, trans_ratio=0.25):
        trans_patch_index = random.randint(0,8)
        transed_img,gt_pre,gt = self.random_pix_trans(imgs,trans_patch_size=self.trans_patch_size[trans_patch_index])
        img_one,img_two = transed_img
        x,y = self.forward_encoder(img_one,img_two)

        c_in_x, c_in_y, common, positive_x, nagetive_x, positive_y, nagetive_y, xy = self.forward_decoder(x,y)  # [N, L, p*p*3]
        # loss = self.forward_loss(imgs, pred, mask)
        c_in_x = self.unpatchify(c_in_x)
        c_in_y = self.unpatchify(c_in_y)
        common = self.unpatchify(common)
        positive_x = self.unpatchify(positive_x)
        nagetive_x = self.unpatchify(nagetive_x)
        positive_y = self.unpatchify(positive_y)
        nagetive_y = self.unpatchify(nagetive_y)
        xy = self.unpatchify(xy)

        return [c_in_x,c_in_y,common,positive_x,nagetive_x,positive_y,nagetive_y,xy],gt_pre,gt

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=1,trans_patch_size=2, embed_dim=144,in_chans=1, depth=8, num_heads=16,
        decoder_embed_dim=144, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

if __name__ == "__main__":
    tensor = torch.ones([1,2,3]).to("cuda")
    tensor_ = nolinear_trans_patch(tensor)
    print(tensor_)