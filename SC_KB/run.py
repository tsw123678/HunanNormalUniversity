import json
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import os
import warnings
import torchvision.datasets as dset
from PIL import Image

warnings.filterwarnings("ignore")
from base_nets import base_net
from channel_nets import channel_net
from neural_nets import CAE, VAE
import time
import numpy as np
import torchvision
import random

# 图像标准化的均值和方差
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
torch.cuda.set_device(0)


class params():
    checkpoint_path = "checkpoints"
    device = "cuda"
    dataset = r"E:\pengyubo\datasets\VOC2012_seg"
    log_path = "logs"
    epoch = 100
    lr = 1e-3
    batchsize = 128
    # snr越高：传输过程中有效信号与噪声比例越大
    snr = 25
    weight_delay = 1e-5
    # 开启语义压缩
    use_SAC = True


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 关闭CuDNN的自动调整策略
    torch.backends.cudnn.benchmark = False
    # 启用CuDNN的确定性模式
    torch.backends.cudnn.deterministic = True


# 将预测的图像数据保存到文件中进行可视化
def show_images(pred_images, filename):
    """
    pred_images: 重建的图像
    filename: 保存地址
    """

    # 从范围[-1, 1]转换为[0, 1]范围的像素值。
    # 将预测的图像数据从模型输出的标准化范围还原为正常的图像像素值范围
    imgs_sample = (pred_images.data + 1) / 2.0
    # 每行显示10张图像
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)


class custom_datasets(Dataset):
    def __init__(self, data):
        self.data = data.imgs
        self.img_transform = self.transform()

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        img = Image.open(self.data[item][0]).convert('RGB')
        img = self.img_transform(img)
        return img, self.data[item][0]

    def transform(self):
        compose = [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
        return transforms.Compose(compose)


# 重建图像与原图损失
# def loss_vae(recon_x, x, mu, logvar):
#     """
#     recon_x: generating images
#     x: origin images
#
#     mu：潜在变量的均值（latent mean），通常代表潜在空间中的数据点的位置
#     logvar：潜在变量的对数方差（log variance），用于表示潜在空间中的数据点的分布
#
#     """
#     mse = F.l1_loss(recon_x, x) * 100  # mse loss
#     KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
#     # 促使潜在变量的分布接近标准正态分布，从而使模型更好地进行数据生成和潜在表示学习
#     KLD = torch.sum(KLD_element).mul_(-0.5)
#     return mse + KLD


def train(model, train_dataloader, arg: params):
    # laod weights
    weights_path = os.path.join(arg.checkpoint_path, f"raw_snr{arg.snr}_SC.pth")
    # weights = torch.load(weights_path, map_location="cpu")
    # model.load_state_dict(weights)

    model = model.to(arg.device)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr,
                                 weight_decay=arg.weight_delay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.1, patience=100,
                                                           verbose=True, threshold=0.0001,
                                                           threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,verbose=True)

    # define loss function
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    # training
    model.train()

    # unsupervised learning
    loss_record = []
    ratio = 2
    for epoch in range(arg.epoch):
        start = time.time()
        losses = []
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            print("图像所占bit", x.element_size() * x.nelement())
            x = x.to(arg.device)

            # unlabel

            encoding, encoding_with_noise, decoding = model(x)
            print("图像所占bit", encoding.element_size() * encoding.nelement())
            # compute MI
            loss_1 = mse(encoding, encoding_with_noise)
            loss_2 = l1(torch.pow((decoding + 1) * ratio, 2), torch.pow((x + 1) * ratio, 2))
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            losses.append(loss.item())
        losses = np.mean(losses)
        loss_record.append(losses)
        print(
            f"epoch {epoch} | MI loss: {loss_1.item()} | Rec loss: {loss_2.item()} | total loss: {loss.item()} | waste time: {time.time() - start}")
        if epoch % 5 == 0:
            os.makedirs(os.path.join(arg.log_path, f"{arg.snr}"), exist_ok=True)
            show_images(x.detach().cpu(), os.path.join(arg.log_path, f"{arg.snr}", "raw_imgs.jpg"))
            show_images(decoding.detach().cpu(), os.path.join(arg.log_path, f"{arg.snr}", "raw_rec_imgs.jpg"))
        with open(os.path.join(arg.log_path, f"raw_snr{arg.snr}_loss.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(loss_record, indent=4, ensure_ascii=False))
        # if (epoch+1)%10==0:
        #     ratio*=1.1
        torch.save(model.state_dict(), weights_path)


# 带有mask
# 只训练ASC
def train_ASC(model, train_dataloader, arg: params):
    # laod weights
    weights_path = os.path.join(arg.checkpoint_path, f"seg_snr{arg.snr}_SC.pth")
    weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights)

    model = model.to(arg.device)
    # define optimizer
    # 训练model中的mask isc_model.Mask.parameters()
    optimizer = torch.optim.Adam(model.isc_model.Mask.parameters(), lr=arg.lr,
                                 weight_decay=arg.weight_delay)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.1, patience=100,
                                                           verbose=True, threshold=0.0001,
                                                           threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,verbose=True)

    # define loss function
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    # training
    model.train()

    # unsupervised learning
    loss_record = []
    ratio = 2
    for epoch in range(arg.epoch):
        start = time.time()
        losses = []
        for i, (x, y) in enumerate(train_dataloader):
            model.zero_grad()

            print("图像所占bit", x.element_size() * x.nelement())
            x = x.to(arg.device)
            # unlabel
            encoding, encoding_with_noise, decoding = model(x)
            print("语义所占bit", encoding.element_size() * encoding.nelement())

            # compute MI
            # 经过信道的差异损失
            loss_1 = mse(encoding, encoding_with_noise)
            # 重建图像和x的差异损失
            loss_2 = l1(torch.pow((decoding + 1) * ratio, 2), torch.pow((x + 1) * ratio, 2))
            loss = loss_1 + loss_2

            loss.backward()

            optimizer.step()
            scheduler.step(loss)

            losses.append(loss.item())

        # 平均损失
        losses = np.mean(losses)
        loss_record.append(losses)

        print(
            f"epoch {epoch} | MI loss: {loss_1.item()} | Rec loss: {loss_2.item()} | total loss: {loss.item()} | waste time: {time.time() - start}")

        if epoch % 5 == 0:
            # logs/25
            os.makedirs(os.path.join(arg.log_path, f"{arg.snr}"), exist_ok=True)
            # 原图与重建图
            show_images(x.detach().cpu(), os.path.join(arg.log_path, f"{arg.snr}", "seg_imgs.jpg"))
            show_images(decoding.detach().cpu(), os.path.join(arg.log_path, f"{arg.snr}", "seg_rec_imgs.jpg"))

        # 记录损失 indent=4：缩进
        with open(os.path.join(arg.log_path, f"seg_snr{arg.snr}_loss.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(loss_record, indent=4, ensure_ascii=False))

        # if (epoch+1)%10==0:
        #     ratio*=1.1
        torch.save(model.state_dict(), weights_path)


import math
from ssim import SSIM


def evaluate(model, test_dataloader, arg):

    # 峰值信噪比（PSNR）
    # 测量图像重建的指标
    def psnr_loss(mse):
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    # SSIM是一种常用的用于评估两幅图像之间结构相似性的指标
    ssim_loss = SSIM().to(arg.device)

    # laod weights
    weights_path = os.path.join(arg.checkpoint_path, f"raw_snr{arg.snr}_SC.pth")
    weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights)
    model = model.to(arg.device)

    # define evaluations
    model.eval()

    psnr_res = []
    ssim_res = []

    for i, (x, y) in enumerate(test_dataloader):
        x = x.to(arg.device)

        # unlabel
        encoding, encoding_with_noise, decoding = model(x)

        # compute MI
        mse = F.mse_loss((decoding + 1) / 2, (x + 1) / 2)
        psnr_res.append(psnr_loss(mse.item()))
        ssim_res.append(ssim_loss(decoding, x).item())

    psnr_res = np.mean(psnr_res)
    ssim_res = np.mean(ssim_res)

    print(
        f"{weights_path} | SNR:{arg.snr} | PSNR: {psnr_res} | SSIM: {ssim_res}")


if __name__ == '__main__':
    same_seeds(1024)
    arg = params()

    Img_data = dset.ImageFolder(root=arg.dataset)
    datasets = custom_datasets(Img_data)
    train_dataloader = DataLoader(dataset=datasets, batch_size=arg.batchsize, shuffle=True, num_workers=0,
                                  drop_last=False)

    SC_model = CAE(input_dim=3, SAC=arg.use_SAC)
    # SC_model = VAE(SAC=arg.use_SAC)
    channel_model = channel_net(M=5408, snr=arg.snr)
    model = base_net(SC_model, channel_model)
    train_ASC(model, train_dataloader, arg)

    # # evaluate
    # for snr in [0,5,10,15,20,25]:
    #     arg.snr = snr
    #     channel_model = channel_net(M=5408, snr=arg.snr)
    #     model = base_net(SC_model, channel_model)
    #     evaluate(model, train_dataloader, arg)
