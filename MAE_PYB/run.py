import nets
import sys
import os
import requests
from torch.utils.data import Dataset, DataLoader
import torchvision
import random
import pickle
import json
from auto_encoder import basic_DNN, DSN
from torchvision.transforms import transforms
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import random
from PIL import Image
from torch import nn

torch.cuda.set_device(1)
random.seed(2)
torch.manual_seed(2)
checkpoints_dir = "checkpoints"
record_dir = "images"
batchsize = 16

# 创建数据集
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
        return transforms.Compose(compose)


data_path = r"D:\pengyubo\pythonProj\single_images\train"
Img_data = dset.ImageFolder(root=data_path)
datasets = custom_datasets(Img_data)
train_dataloader = DataLoader(dataset=datasets, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=True)

data_path = r"D:\pengyubo\pythonProj\single_images\valid"
Img_data = dset.ImageFolder(root=data_path)
datasets = custom_datasets(Img_data)
val_dataloader = DataLoader(dataset=datasets, batch_size=1, shuffle=False, num_workers=0)

data_path = r"D:\pengyubo\pythonProj\single_images\test"
Img_data = dset.ImageFolder(root=data_path)
datasets = custom_datasets(Img_data)
test_dataloader = DataLoader(dataset=datasets, batch_size=batchsize, shuffle=False, num_workers=0)


# 创建模型
def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    # 动态加载nets.py中的mae_vit_large_patch16作为model
    model = getattr(nets, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    return model


channel = "rali"
snr = 25
channel_model = basic_DNN(snr=snr, rali=True if channel == "rali" else False)
channel_model.to("cuda")

DSN_model = DSN().to("cuda")
# chkpt_dir = 'mae_visualize_vit_large.pth'
# model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
chkpt_dir = 'mae_visualize_vit_large_ganloss.pth'
model_mae = prepare_model('mae_visualize_vit_large_ganloss.pth', 'mae_vit_large_patch16')
model_mae.eval()
print('Model loaded.')


def show_images(bs_images, pred_images, fname, im_masked=None):
    imgs_sample = (bs_images.data + 1) / 2.0
    filename = fname + "origin_images.jpg"
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)

    # Show 32 of the images.
    # grid_img = torchvision.utils.make_grid(imgs_sample[:100].cpu(), nrow=10)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(grid_img.permute(1, 2, 0))
    # print("origin images")
    # plt.show()

    imgs_sample = (pred_images.data + 1) / 2.0
    filename = fname + "pred_images.jpg"
    torchvision.utils.save_image(imgs_sample, filename, nrow=1)

    # Show 32 of the images.
    # grid_img = torchvision.utils.make_grid(imgs_sample[:100].cpu(), nrow=10)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(grid_img.permute(1, 2, 0))
    # plt.title("reconstruct images")
    # plt.show()

    # masked image
    if im_masked != None:
        imgs_sample = (im_masked.data + 1) / 2.0
        filename = fname + f"mask_{mask_ratio}_images.jpg"
        torchvision.utils.save_image(imgs_sample, filename, nrow=10)

        # Show 32 of the images.
        grid_img = torchvision.utils.make_grid(imgs_sample[:100].cpu(), nrow=10)
        plt.figure(figsize=(20, 20))
        plt.imshow(grid_img.permute(1, 2, 0))
        print("mask images")

    # ### Compress the generated images using **tar**.
    # Save the generated images.
    # os.makedirs('output', exist_ok=True)
    # for i in range(len(imgs_sample)):
    #     torchvision.utils.save_image(imgs_sample[i], f'output/{i + 1}.jpg')


# 训练参数设置
epoch_num = 10
lr = 1e-3
optimizer_auto = torch.optim.Adam(channel_model.parameters(), lr=lr, weight_decay=1e-5)
optimizer_DSN = torch.optim.Adam(DSN_model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_auto, mode='min', factor=0.1, patience=50,
                                                       verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                       min_lr=0, eps=1e-08)
mask_ratio = 0  # 遮挡比例
criticisen = torch.nn.MSELoss()


# train
def train_with_no_DSN():
    # channel_model = basic_DNN()
    # channel_model.to("cuda")
    # channel_model.load_state_dict(torch.load(checkpoints_dir+"/AiT_rali_snr15_checkpoint.pth", map_location=lambda storage, loc: storage))
    g_loss_record = []
    show_pic_round = 500
    steps = 0
    for epoch in range(epoch_num):
        epoch_loss = []
        channel_model.train()
        for batch_img, _ in train_dataloader:
            optimizer_auto.zero_grad()
            # batch_img = torch.einsum('nhwc->nchw', batch_img)
            # run MAE encoder
            latent, mask, ids_restore = model_mae(batch_img, mask_ratio=mask_ratio)
            latent = latent.to("cuda")
            # run channel model
            ouput = channel_model(latent)
            loss = criticisen(latent, ouput) * 100
            epoch_loss.append(loss)
            loss.backward()
            optimizer_auto.step()
            g_loss_record.append(loss.item())
            print(f"epoch{epoch} train loss:{loss.item()}")
            # run MAE decoder
            if steps % show_pic_round == 0:
                _, pred_imgs, _ = model_mae(batch_img, [ouput.detach().cpu(), mask, ids_restore])
                # x = torch.einsum('nchw->nhwc', batch_img)
                x = batch_img
                y = model_mae.unpatchify(pred_imgs)
                # y = torch.einsum('nchw->nhwc', y).detach().cpu()
                show_images(x, y, f"{checkpoints_dir}/AwD_{channel}_train_{steps}_")
            steps += 1
            scheduler.step(loss)
        # channel_model.eval()
        # for batch_img in val_dataloader:
        #     with torch.no_grad():
        #         # batch_img = torch.einsum('nhwc->nchw', batch_img)
        #         # run MAE encoder
        #         latent, mask, ids_restore = model_mae(batch_img, mask_ratio=0.0)
        #         latent = latent.to("cuda")
        #         # run channel model
        #         ouput = channel_model(latent)
        #         loss = criticisen(latent,ouput)
        #         print(f"epoch{epoch} val loss:{loss.item()}")
        #         # run MAE decoder
        #         _, pred_imgs, _ = model_mae(batch_img,[ouput.detach().cpu(), mask, ids_restore])
        #         # x = torch.einsum('nchw->nhwc', batch_img)
        #         x = batch_img
        #         y = model_mae.unpatchify(pred_imgs)
        #         # y = torch.einsum('nchw->nhwc', y).detach().cpu()
        #         show_images(x,y,f"{record_dir}/val_{epoch}_")

        torch.save(channel_model.state_dict(), f"{checkpoints_dir}/AwD_{channel}_snr{snr}_checkpoint.pth")
        with open(f"AwD_{channel}_loss.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(g_loss_record, indent=4, ensure_ascii=False))


def train_with_DSN():
    # channel_model.load_state_dict(torch.load(checkpoints_dir+"/AiT_rali_snr15_checkpoint.pth", map_location=lambda storage, loc: storage))
    # DSN_model.load_state_dict(torch.load(checkpoints_dir+"/DSN_rali_snr15_checkpoint.pth", map_location=lambda storage, loc: storage))
    show_pic_round = 20
    steps = 0
    first_itreation = True
    g_loss_r = []
    d_loss_r = []
    skip_step = 1
    loss_record = []
    for epoch in range(epoch_num):
        for batch_img, _ in train_dataloader:
            # train DSN
            if steps % skip_step == 0:
                channel_model.eval()
                DSN_model.train()
                optimizer_DSN.zero_grad()

                latent, mask, ids_restore = model_mae(batch_img, mask_ratio=mask_ratio)
                latent = latent.to("cuda")
                # run channel model
                ouput = channel_model(latent)

                # run MAE decoder
                _, pred_imgs, mask = model_mae(batch_img, [ouput.detach().cpu(), mask, ids_restore])
                real_imgs = batch_img.to("cuda")
                fake_imgs = model_mae.unpatchify(pred_imgs).to("cuda")
                pred_real = DSN_model(real_imgs)
                pred_fake = DSN_model(fake_imgs)
                if first_itreation:
                    first_itreation = False
                    r_label = torch.ones(pred_real.shape).cuda()
                    f_label = torch.zeros(pred_fake.shape).cuda()

                d_loss_real = F.binary_cross_entropy_with_logits(pred_real, r_label)

                d_loss_fake = F.binary_cross_entropy_with_logits(pred_fake, f_label)

                d_loss = d_loss_real + d_loss_fake
                d_loss_r.append(d_loss.item())
                if len(d_loss_r) > 10:
                    d_loss_r.pop(0)
                print(f"epoch {epoch},steps{steps}, d_loss:{d_loss.item()}")
                d_loss.backward()
                optimizer_DSN.step()

            ###### train G
            channel_model.train()
            DSN_model.eval()
            optimizer_auto.zero_grad()
            # batch_img = torch.einsum('nhwc->nchw', batch_img)
            # run MAE encoder
            latent, mask, ids_restore = model_mae(batch_img, mask_ratio=mask_ratio)
            latent = latent.to("cuda")
            # run channel model
            ouput = channel_model(latent)
            g_loss_mu = F.mse_loss(latent, ouput)

            _, pred_imgs, mask = model_mae(batch_img, [ouput.detach().cpu(), mask, ids_restore])
            fake_imgs = model_mae.unpatchify(pred_imgs)
            pred_fake = DSN_model(fake_imgs.to("cuda"))
            g_loss_L1 = F.mse_loss(fake_imgs, batch_img)
            g_loss_d = F.binary_cross_entropy_with_logits(pred_fake, r_label)

            g_loss = g_loss_d + g_loss_L1 * 50 + g_loss_mu * 50
            print(g_loss_d.item(), g_loss_L1.item(), g_loss_mu.item())
            g_loss_r.append(g_loss_d.item())
            if len(g_loss_r) > 10:
                g_loss_r.pop(0)
            g_loss.backward()
            optimizer_auto.step()
            scheduler.step(g_loss)
            print(f"epoch{epoch},steps{steps}, train g_loss:{g_loss.item()}")
            loss_record.append(g_loss.item())
            if len(g_loss_r) == 10 and len(d_loss_r) == 10:
                if np.mean(g_loss_r) / np.mean(d_loss_r) > 2:
                    skip_step = skip_step + 1 if skip_step < 10 else 10
                else:
                    skip_step = 1

            if steps % show_pic_round == 0:
                _, pred_imgs, _ = model_mae(batch_img, [ouput.detach().cpu(), mask, ids_restore])
                # x = torch.einsum('nchw->nhwc', batch_img)
                x = batch_img
                y = model_mae.unpatchify(pred_imgs)
                # y = torch.einsum('nchw->nhwc', y).detach().cpu()
                show_images(x, y, f"{record_dir}/AiT_{channel}_train_{steps}_")
            steps += 1

        # channel_model.eval()
        # for batch_img,_ in val_dataloader:
        #     with torch.no_grad():
        #         # batch_img = torch.einsum('nhwc->nchw', batch_img)
        #         # run MAE encoder
        #         latent, mask, ids_restore = model_mae(batch_img, mask_ratio=mask_ratio)
        #         latent = latent.to("cuda")
        #         # run channel model
        #         ouput = channel_model(latent)
        #         # loss = criticisen(latent,ouput)
        #         # print(f"epoch{epoch} val loss:{loss.item()}")
        #         # run MAE decoder
        #         _, pred_imgs, _ = model_mae(batch_img,[ouput.detach().cpu(), mask, ids_restore])
        #         # x = torch.einsum('nchw->nhwc', batch_img)
        #         x = batch_img
        #         y = model_mae.unpatchify(pred_imgs)
        #         # y = torch.einsum('nchw->nhwc', y).detach().cpu()
        #         show_images(x,y,f"{record_dir}/val_{epoch}")

        torch.save(channel_model.state_dict(), f"{checkpoints_dir}/AiT_{channel}_snr{snr}_checkpoint.pth")
        torch.save(DSN_model.state_dict(), f"{checkpoints_dir}/DSN_{channel}_snr{snr}_checkpoint.pth")
        with open(f"AiT_{channel}_loss.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(loss_record, indent=4, ensure_ascii=False))


# test
def test():
    channel_model = basic_DNN()
    channel_model.to("cuda")

    DSN_model = DSN().to("cuda")
    channel_model.load_state_dict(
        torch.load(checkpoints_dir + "/14_checkpoint.pth", map_location=lambda storage, loc: storage))
    channel_model.eval()
    for i, batch_img in enumerate(test_dataloader):
        with torch.no_grad():
            # batch_img = torch.einsum('nhwc->nchw', batch_img)
            # run MAE encoder
            latent, mask, ids_restore = model_mae(batch_img, mask_ratio=mask_ratio)
            latent = latent.to("cuda")
            # run channel model
            ouput = channel_model(latent)
            loss = criticisen(latent, ouput)
            print(f"test loss:{loss.item()}")
            # run MAE decoder
            _, pred_imgs, mask = model_mae(batch_img, [ouput.detach().cpu(), mask, ids_restore])
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, model_mae.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
            mask = model_mae.unpatchify(mask)  # 1 is removing, 0 is keeping
            im_masked = batch_img * (1 - mask)
            # x = torch.einsum('nchw->nhwc', batch_img)
            x = batch_img
            y = model_mae.unpatchify(pred_imgs)
            # y = torch.einsum('nchw->nhwc', y).detach().cpu()
            show_images(x, y, f"{record_dir}/test_{i}_", im_masked)


# 获取文件夹下所有的图片文件路径
def getallpics(path, piclist):
    for file in os.listdir(path):
        # print(file,filecount)
        if file.lower().endswith('.jpg') or file.lower().endswith('.png') or file.lower().endswith(
                '.jpeg') or file.lower().endswith('.bmp'):
            piclist.append(os.path.join(path, file))
        elif os.path.isdir(os.path.join(path, file)):
            getallpics(os.path.join(path, file), piclist)
        else:
            continue


def AiT_exert_recon_images(src_path, dst_path, snr, channel_type, dataloader):
    channel_model = basic_DNN(snr=snr, rali=True if channel_type == "rali" else False)
    channel_model.to("cuda")
    channel_model.load_state_dict(
        torch.load(checkpoints_dir + f"/AwD_{channel_type}_snr25_checkpoint.pth",
                   map_location=lambda storage, loc: storage))
    channel_model.train()
    with torch.no_grad():
        for batch_img, src_pic in dataloader:
            # src_img = cv2.imread(src_pic)
            # batch_img = torch.from_numpy(src_img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            latent, mask, ids_restore = model_mae(batch_img, mask_ratio=0.0)
            latent = latent.to("cuda")
            # run channel model
            ouput = channel_model(latent)

            # run MAE decoder
            _, pred_imgs, _ = model_mae(batch_img, [ouput.detach().cpu(), mask, ids_restore])
            y = model_mae.unpatchify(pred_imgs)
            imgs_sample = (y.data + 1) / 2.0
            save_pth = src_pic[0].replace(src_path, dst_path)
            os.makedirs(os.path.sep.join(save_pth.split(os.path.sep)[:-1]), exist_ok=True)
            torchvision.utils.save_image(imgs_sample, save_pth, nrow=1)
            print(src_pic, "->", save_pth)


import cv2
import os


def exrt():
    src_path = r"D:\pengyubo\pythonProj\single_images\valid"
    snr_list = [0, 5, 10, 15, 20, 25]
    Img_data = dset.ImageFolder(root=src_path)
    datasets = custom_datasets(Img_data)
    dataloader = DataLoader(dataset=datasets, batch_size=1, shuffle=False, num_workers=1)
    for snr in snr_list:
        for channel in ["rali", "awgn"]:
            print(f"channel {channel}, snr {snr}")
            dst_path = fr"D:\pengyubo\pythonProj\single_images\AwD_{channel}_{snr}_res"
            AiT_exert_recon_images(src_path, dst_path, snr, channel, dataloader)


if __name__ == '__main__':
    train_with_DSN()
    # train_with_no_DSN()
    # test()
    # exrt()
