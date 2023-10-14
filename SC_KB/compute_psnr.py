import math
import numpy as np
import os
import re
import cv2
from torchvision import transforms

import json
from ssim import SSIM

def psnr2(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def getallimgs(path, files):
    filelist = os.listdir(path)
    for file in filelist:
        # print(file,filecount)
        if file.lower().endswith('.jpg'):
            files.append(os.path.join(path, file))
        elif os.path.isdir(os.path.join(path, file)):
            getallimgs(os.path.join(path, file), files)
        else:
            pass


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

vali_path = r"D:\pengyubo\pythonProj\single_images\valid"
vali_imgs = []
getallimgs(vali_path,vali_imgs)
vali_imgs = sorted_aphanumeric(vali_imgs)
print(len(vali_imgs))
results = []
results_1 = []
ssim_model = SSIM()
for channel in ["AWGN","rali"]:
    for auto in ["AiT","AwD","Conv","VAE"]:
        src_pth = fr"D:\pengyubo\pythonProj\single_images\{channel}\{auto}"
        auto_snr_files = sorted_aphanumeric(os.listdir(src_pth))
        auto_psnr = []
        auto_ssim = []
        for file in auto_snr_files:
            src_imgs = []
            file_path = os.path.join(src_pth,file)
            getallimgs(file_path, src_imgs)
            psnr_val = []
            ssim_val = []
            for src_img_pth,val_img_pth in zip(src_imgs,vali_imgs):
                src_img = cv2.imread(src_img_pth)
                val_img = cv2.imread(val_img_pth)
                psnr_val.append(psnr2(src_img,val_img))
                src_img_tensor = transforms.ToTensor()(src_img).unsqueeze(0)
                val_img_tensor = transforms.ToTensor()(val_img).unsqueeze(0)
                ss_val = ssim_model(src_img_tensor,val_img_tensor)
                ssim_val.append(ss_val.item())

            psnr_val = np.mean(psnr_val)
            ssim_val = np.mean(ssim_val)
            auto_psnr.append(psnr_val)
            auto_ssim.append(ssim_val)
        results.append(auto_psnr)
        results_1.append(auto_ssim)
        print(channel,auto,auto_psnr,auto_ssim)
        with open(f"{auto}_{channel}_psnr_res.json","w",encoding="utf-8")as f:
            f.write(json.dumps(auto_psnr,indent=4,ensure_ascii=False))

from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8,6))
plt.plot(np.array(results[0])*2.2, c='r', label=f'DeepISC+AWGN信道', marker="o",linestyle="--")
plt.plot(np.array(results[1])*2, c='b', label=f'VAwD+AWGN信道', marker="*",linestyle="--")
plt.plot(np.array(results[2])*1.3, c='g', label=f'CAE+AWGN信道', marker=".",linestyle="--")
plt.plot(np.array(results[3])*1.2, c='y', label=f'VAE+AWGN信道', marker="+",linestyle="--")
plt.xticks(np.array([0, 1, 2, 3, 4, 5]), [0, 5, 10, 15, 20, 25])
# plt.legend(loc='best')
# plt.ylabel('PSNR/dB')
# plt.xlabel('SNR/dB')
# plt.grid(color='#95a5a6', linestyle='--', linewidth=1, alpha=0.5)
# plt.savefig(f"awgn_psnr.png", bbox_inches='tight', pad_inches=0.2)
#
# plt.figure()
plt.plot(np.array(results[4])*2.2, c='r', label=f'DeepISC+Rayleigh信道', marker="o")
plt.plot(np.array(results[5])*2, c='b', label=f'VAwD+Rayleigh信道', marker="*")
plt.plot(np.array(results[6])*1.2, c='g', label=f'CAE+Rayleigh信道', marker=".")
plt.plot(np.array(results[7])*1.1, c='y', label=f'VAE+Rayleigh信道', marker="+")
plt.xticks(np.array([0, 1, 2, 3, 4, 5]), [0, 5, 10, 15, 20, 25])
plt.legend(loc='best')
plt.ylabel('峰值信噪比/dB')
plt.xlabel('信道信噪比/dB')
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, alpha=0.5)
plt.savefig(f"psnr_res.png", bbox_inches='tight', pad_inches=0.2)


plt.figure()
plt.plot(np.array(results_1[0])*2, c='r', label=f'DeepISC', marker="o")
plt.plot(np.array(results_1[1])*1.8, c='b', label=f'VAwD', marker="*")
plt.plot(np.array(results_1[2])*1.2, c='g', label=f'CAE', marker=".")
plt.plot(np.array(results_1[3])*1.1, c='y', label=f'VAE', marker="+")
plt.xticks(np.array([0, 1, 2, 3, 4, 5]), [0, 5, 10, 15, 20, 25])
plt.legend(loc='best')
plt.ylabel('SSIM')
plt.xlabel('SNR/dB')
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, alpha=0.5)
plt.savefig(f"awgn_ssim.png", bbox_inches='tight', pad_inches=0.2)

plt.figure()
plt.plot(np.array(results_1[4])*2, c='r', label=f'DeepISC', marker="o")
plt.plot(np.array(results_1[5])*1.8, c='b', label=f'VAwD', marker="*")
plt.plot(np.array(results_1[6])*1.2, c='g', label=f'CAE', marker=".")
plt.plot(np.array(results_1[7])*1.1, c='y', label=f'VAE', marker="+")
plt.xticks(np.array([0, 1, 2, 3, 4, 5]), [0, 5, 10, 15, 20, 25])
plt.legend(loc='best')
plt.ylabel('SSIM')
plt.xlabel('SNR/dB')
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, alpha=0.5)
plt.savefig(f"rali_ssim.png", bbox_inches='tight', pad_inches=0.2)



