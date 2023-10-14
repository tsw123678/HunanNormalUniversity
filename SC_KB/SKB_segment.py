import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def show_anns(anns,dst_img_path):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 255
    for ann in sorted_anns[:int(len(sorted_anns)/5)]:
        m = ann['segmentation']
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        color_mask = np.array([0,0,0,0])
        img[m] = color_mask
    ax.imshow(img)

    dirs = os.path.sep.join(dst_img_path.split(os.path.sep)[:-1])
    print(dirs)
    os.makedirs(dirs,exist_ok=True)
    plt.axis('off')
    plt.savefig(dst_img_path,bbox_inches='tight', pad_inches=0)

def show_mask(mask, ax, image,dst_img_path):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * image
    ax.imshow(mask_image)
    dirs = os.path.sep.join(dst_img_path.split(os.path.sep)[:-1])
    os.makedirs(dirs, exist_ok=True)
    plt.axis('off')
    # dst_img_path = dst_img_path.replace(".jpg",f"_{id}.jpg")
    plt.savefig(dst_img_path, bbox_inches='tight', pad_inches=0)

def image_segment(dataset="",device="cuda"):
    sam_checkpoint = "test_SAM/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamPredictor(sam)
    # mask_generator = SamAutomaticMaskGenerator(model=sam,
    #                                            points_per_side=32,
    #                                            pred_iou_thresh=0.8,
    #                                            stability_score_thresh=0.9,
    #                                            crop_n_layers=1,
    #                                            crop_n_points_downscale_factor=2,
    #                                            min_mask_region_area=100, )



    for img in os.listdir(dataset):
        if not img.endswith(".jpg"):
            continue
        img_path = os.path.join(dataset,img)
        dst_img_path = img_path.replace("VOC2012_source","VOC2012_seg")
        if os.path.exists(dst_img_path):
            continue
        box_json_path = img_path.replace(".jpg",".json")
        with open(box_json_path,"r",encoding="utf-8")as f:
            boxes = json.load(f)
        if boxes == []:
            continue

        input_boxes = torch.tensor(boxes, device=device)

        x_min = torch.min(input_boxes[:,0]).cpu()
        y_min = torch.min(input_boxes[:,1]).cpu()
        x_max = torch.max(input_boxes[:,2]).cpu()
        y_max = torch.max(input_boxes[:,3]).cpu()
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_generator.set_image(image)
        transformed_boxes = mask_generator.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = mask_generator.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        # masks = mask_generator.generate(image)
        merge_mask = masks[0]
        for mask in masks:
            merge_mask = merge_mask|mask
        plt.figure()
        plt.imshow(image)
        # print(x_min,x_max,y_min,y_max)
        plt.xlim((x_min,x_max))
        plt.ylim((y_max,y_min))
        show_mask(merge_mask.cpu().numpy(), plt.gca(), image,dst_img_path)
        # show_anns(masks,dst_img_path)
        print(img_path,"process success!")
        plt.close()

if __name__ == '__main__':
    dataset = r"D:\pengyubo\datasets\VOC2012_source\VOC2012"
    image_segment(dataset,"cuda")
