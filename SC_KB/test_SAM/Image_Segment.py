import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import copy

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_interesting_object(mask, image, ax):
    print(mask.shape)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1)*image
    mean = 0
    sigma = 100
    # print("this is train sigma")
    # print(sigma)

    # np.random.normal(loc=mean,scale=sigma,size=(row,col,ch))生成高斯分布的概率密度随机数，均值，方差，输入样式
    # 建立方差随机的高斯噪声，方差分布在0到50之间
    gauss = np.random.normal(mean, sigma, (h, w, 3))
    gauss = gauss.reshape(h, w, 3)
    gauss_2 = np.random.normal(mean, 50, (h, w, 3))
    gauss_2 = gauss_2.reshape(h, w, 3)
    gauss[mask_image!=0]=0
    gauss_2[mask_image==0] = 0
    noisy = mask_image + gauss_2
    # np.clip(a,a_min,a_max)将输入的原始a限制在a_min与a_max之间，小于a_min赋值为a_min,大于a_max赋值为a_max
    noisy = np.clip(noisy, 0, 255)
    # noise.astype('uint8'),转变数据类型为uint8型，uint8为8位无符号整数类型，表示范围为[0:255]
    noisy = noisy.astype('uint8')

    # ax.imshow(mask_image)
    ax.imshow(noisy)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

image = cv2.imread('images/2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()

import sys
sys.path.append("../..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)
plt.imshow(image)
count  = 1
while True:
    select_points = input("依次输入坐标值，空格隔开：")
    if select_points == "":
        break
    select_points = select_points.split(" ")
    select_points = np.array([int(val) for val in select_points]).reshape(-1,2)
    print(select_points)

    input_label = np.array([1 for i in range(select_points.shape[0])])
    print(input_label)

    masks, _, _ = predictor.predict(
        point_coords=select_points,
        point_labels=input_label,
        multimask_output=False,
    )

    plt.figure(figsize=(10,10))
    show_interesting_object(masks,image,plt.gca())
    # show_mask(masks, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig(f"res_{count}.png",bbox_inches='tight', pad_inches=0)
    plt.show()
    count += 1