# -*- coding: utf-8 -*-
import torch.nn as nn
import network
from ssim import *
# from ssim_fuyu import *
import torch
import torch.optim as optim
import torchvision
import os
import torch.nn.functional as F
from contiguous_params import ContiguousParams
import numpy as np
from utils import gradient,gradient_sobel,gradient_Isotropic

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.fusion = network.__dict__['mae_vit_large_patch16'](norm_pix_loss=False)
        # self.fusion = cnn_network.__dict__['mae_cnn_large_patch16'](norm_pix_loss=False)
        self.MSE_fun = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.SSIM_fun = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1, K=(0.01,0.03))

        # self.SSIM_fun = SSIM()
        if args.contiguousparams == True:
            print("ContiguousParams---")
            parametersF = ContiguousParams(self.fusion.parameters())
            self.optimizer_G = optim.Adam(parametersF.contiguous(), lr=args.lr)
        else:
            self.optimizer_G = optim.Adam(self.fusion.parameters(), lr=args.lr)

        # self.optimizer_G = optim.Adam(self.fusion.parameters(), lr=args.lr)
        self.loss = torch.zeros(1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.5,
                                                                    patience=2,
                                                                    verbose=False, threshold=0.0001,
                                                                    threshold_mode='rel',
                                                                    cooldown=0, min_lr=0, eps=1e-10)
        self.min_loss = 1000
        self.mean_loss = 0
        self.count_batch = 0
        self.args = args
        if args.multiGPU:
            self.mulgpus()
        self.load()
        # [a*pixel+b*ssim+c*gradient]
        self.loss_hyperparameters = [1, 300, 10, 100]

    def load(self, ):
        start_epoch = 0
        if self.args.load_pt:
            print("=========LOAD WEIGHTS=========")
            print(self.args.weights_path)
            checkpoint = torch.load(self.args.weights_path)
            start_epoch = checkpoint['epoch'] + 1
            try:
                if self.args.multiGPU:
                    print("load G")
                    self.fusion.load_state_dict(checkpoint['weight'])
                else:
                    print("load G single")
                    # 单卡模型读取多卡模型
                    state_dict = checkpoint['weight']
                    # create new OrderedDict that does not contain `module.`
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace('module.', '')  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.fusion.load_state_dict(new_state_dict)
            except:
                model = self.fusion
                print("weights not same ,try to load part of them")
                model_dict = model.state_dict()
                pretrained = torch.load(self.args.weights_path)['weight']
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in model_dict.items() if k in pretrained}
                left_dict = {k for k, v in model_dict.items() if k not in pretrained}
                print(left_dict)
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                model.load_state_dict(model_dict)
                print(len(model_dict), len(pretrained_dict))
                # model_dict = self.fusion.state_dict()
                # pretrained_dict = {k: v for k, v in model_dict.items() if k in checkpoint['weight'] }
                # print(len(checkpoint['weight'].items()), len(pretrained_dict), len(model_dict))
                # model_dict.update(pretrained_dict)
                # self.fusion.load_state_dict(model_dict)
            print("start_epoch:", start_epoch)
            print("=========END LOAD WEIGHTS=========")
        print("========START EPOCH: %d=========" % start_epoch)
        self.start_epoch = start_epoch

    def forward(self):
        self.img_re,self.gt_pre,self.gt = self.fusion(self.img,self.saliency_map)

    def backward(self):
        # [c_in_x, c_in_y, common, trans_c_in_x, trans_c_in_y, xy], [c_in_x,c_in_y],[gt_x,gt_y,gt_common]
        c_in_x = self.img_re[0]
        c_in_y = self.img_re[1]
        common_re = self.img_re[2]
        positive_x = self.img_re[3]
        nagetive_x = self.img_re[4]
        positive_y = self.img_re[5]
        nagetive_y = self.img_re[6]
        img_re = self.img_re[7]

        gt_c_in_x = self.gt_pre[0]
        gt_c_in_y = self.gt_pre[1]
        # gt_x_one,gt_x_two,gt_y_one,gt_y_two,gt_common
        gt_p_x = self.gt[0]
        gt_n_x = self.gt[1]
        gt_p_y = self.gt[2]
        gt_n_y = self.gt[3]
        gt_common = self.gt[4]

        # gt_xy = self.gt[5]
        gt_xy = self.img
        # 计算ssim损失
        ssim_loss = 1 - self.SSIM_fun(img_re, gt_xy)
        # 计算像素损失
        pixel_loss = self.MSE_fun(img_re, gt_xy)

        DE_loss = self.MSE_fun(c_in_x,gt_c_in_x) + self.MSE_fun(c_in_y,gt_c_in_y) + self.MSE_fun(common_re,gt_common)+\
                  self.MSE_fun(positive_x,gt_p_x) + self.MSE_fun(nagetive_x,gt_n_x)+self.MSE_fun(positive_y,gt_p_y) + self.MSE_fun(nagetive_y,gt_n_y)

        # g_loss = self.MSE_fun(gradient(img_xy),gradient(gt_xy))
        # 损失求和 回传
        loss = self.loss_hyperparameters[0]*pixel_loss+\
               self.loss_hyperparameters[1]*ssim_loss+\
               self.loss_hyperparameters[2]*DE_loss
               # self.loss_hyperparameters[3]*g_loss

        self.optimizer_G.zero_grad()
        loss.backward()
        self.loss = loss
        self.ssim_loss = ssim_loss
        self.pixel_loss = pixel_loss
        self.DE_loss = DE_loss
        # self.g_loss = g_loss
        self.img_re = img_re
        self.positive_x = positive_x
        self.nagetive_x = nagetive_x
        self.positive_y = positive_y
        self.nagetive_y = nagetive_y
        self.common_re = common_re
        self.gt_p_x = gt_p_x
        self.gt_n_x = gt_n_x
        self.gt_p_y = gt_p_y
        self.gt_n_y = gt_n_y
        self.gt_common = gt_common
        # self.img_clear = img_clear




    def mulgpus(self):
        self.fusion = nn.DataParallel(self.fusion.cuda(), device_ids=self.args.GPUs, output_device=self.args.GPUs[0])
        self.D = nn.DataParallel(self.D.cuda(), device_ids=self.args.GPUs, output_device=self.args.GPUs[0])

    def setdata(self, data):
        img,saliency_map = data
        img = img.to(self.args.device)
        self.img = img
        saliency_map = saliency_map.to(self.args.device)
        self.saliency_map = saliency_map

    def step(self):
        self.forward()
        self.backward()  # calculate gradients for G
        self.optimizer_G.step()  # update G weights
        self.count_batch += 1
        self.print = 'Loss: ALL[%.5lf]mean[%.5f] {pixel[%.5lf]ssim[%.5lf]DE[%.5f]}' % \
                     (self.loss.item(),
                      self.mean_loss/self.count_batch,
                      self.pixel_loss.item()*self.loss_hyperparameters[0],
                      self.ssim_loss.item()*self.loss_hyperparameters[1],
                      self.DE_loss.item() * self.loss_hyperparameters[2]
                      )
        self.mean_loss += self.loss.item()

    def saveimg(self, epoch, num=0):
        img_ori = self.img[0].cpu()
        # img_ori = self.gt[5][0].cpu()
        img_re = self.img_re[0].cpu()
        positive_x = self.positive_x[0].cpu()
        nagetive_x = self.nagetive_x[0].cpu()
        positive_y = self.positive_y[0].cpu()
        nagetive_y = self.nagetive_y[0].cpu()
        common_re = self.common_re[0].cpu()

        gt_p_x = self.gt_p_x[0].cpu()
        gt_n_x = self.gt_n_x[0].cpu()
        gt_p_y = self.gt_p_y[0].cpu()
        gt_n_y = self.gt_n_y[0].cpu()
        gt_common = self.gt_common[0].cpu()



        img = torchvision.utils.make_grid([img_re, img_ori,positive_x,gt_p_x,nagetive_x,gt_n_x,positive_y,gt_p_y,nagetive_y,gt_n_y,common_re,gt_common], nrow=2)
        torchvision.utils.save_image(img,
                                     fp=(os.path.join('output/output_guide_decoder/result_%d_%d.jpg' % (epoch, num))))

    # def saveimgfuse(self,name=''):
    #     # self.img_down = self.downsample(self.img)
    #     # self.img_g = gradient(self.img)
    #
    #     img = torchvision.utils.make_grid(
    #         [self.img[0].cpu(), self.img_g[0].cpu(), ((self.g1+self.g2+self.g3)*1.5)[0].cpu()], nrow=3)
    #     torchvision.utils.save_image(img, fp=(os.path.join(name.replace('Test','demo'))))
    #     # torchvision.utils.save_image(img, fp=(os.path.join('output/epoch/'+str(num)+'.jpg')))

    def save(self, epoch):
        ## 保存模型和最佳模型
        self.mean_loss = self.mean_loss / self.count_batch
        if self.min_loss > self.mean_loss:
            torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_fusion.pt'))
            # torch.save({'weight': self.D.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_D.pt'))
            print('[%d] - Best model is saved -' % (epoch))
            print('mean loss :{%.5f} min:{%.5f}' % (self.mean_loss, self.min_loss))
            self.min_loss = self.mean_loss

        if epoch % 1 == 0:
            torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },
                       os.path.join('weights/epoch' + str(epoch) + '_fusion.pt'))
            # torch.save({'weight': self.D.state_dict(), 'epoch': epoch, },os.path.join('weights/epoch' + str(epoch) + '_D.pt'))
        self.mean_loss = 0
        self.count_batch = 0



