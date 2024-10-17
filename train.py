# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from model import Model
from utils import Dataset
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', default=r'/data/Disk_B/MSCOCO2014/train2014', type=str, help='')
    parser.add_argument('--data', default=r'D:\dataset\train2014', type=str, help='')
    parser.add_argument('--load_pt', default=False, type=bool, help='')
    parser.add_argument('--weights_path', default='weights/best_fusion.pt', type=str, help='')
    parser.add_argument('--lr', default= 1e-4, type=float, help='')
    parser.add_argument('--devices', default="0", type=str, help='')
    parser.add_argument('--device', default="cuda", type=str, help='')
    parser.add_argument('--batch_size', default=2, type=int, help='')
    parser.add_argument('--epochs', default=10, type=int, help='')
    parser.add_argument('--multiGPU', default=False, type=bool, help='')
    parser.add_argument('--GPUs', default=[0, 1], type=list, help='')
    parser.add_argument('--backends', default=False, type=bool, help='')
    parser.add_argument('--contiguousparams', default=False, type=bool, help='')
    parser.add_argument('--workers', default=0, type=int, help='')
    return parser.parse_args()




if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    print("===============Initialization===============")
    args = parse_args()
    print(args)
    os.chdir(r'./')
    torch.backends.cudnn.benchmark = args.backends
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    print("epochs", args.epochs, "batch_size", args.batch_size)
    # Dataset
    data = Dataset(args.data, resize= [224,224],  gray = True)
    loader = DataLoader(data, batch_size = args.batch_size, shuffle=True, num_workers=args.workers)

    print("Load ---- {} images".format(len(data)))
    # Model
    model = Model(args).to(args.device)
    # # 这两行用于保存onnx模型文件，用netron打开
    # dummy_input = torch.randn(1,1,256,256,device='cuda')
    # torch.onnx.export(model.fusion,dummy_input,"onnx/2.onnx",verbose=True)
    print("over=======================")
    # 这两行可以打印模型的信息
    # from torchsummary import summary
    # summary(model.fusion, (1, 256, 256), device="cuda")

    start_epoch = model.start_epoch

    # Training
    print('============ Training Begins [epochs:{} == Loss_balance:{}] ==============='.format(args.epochs,model.loss_hyperparameters))

    loss = torch.zeros(1)
    total = sum([params.nelement() for params in model.fusion.parameters()])
    print("Number of params: {%.2f M}" % (total / 1e6))

    for epoch in range(start_epoch,args.epochs):
        model.scheduler.step(loss.item())
        tqdms = tqdm(loader)
        num = 0
        for img in tqdms:
            model.setdata(img)
            model.step()
            tqdms.set_description("epoch:%d %s" % (epoch, model.print))
            num += 1
            if num%1000== 0:
                model.saveimg(epoch, num)
            if num % 1000 == 0:
                model.save(epoch)
        if epoch % 1 == 0:
            model.saveimg(epoch)
        model.save(epoch)
