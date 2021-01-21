import gc
import os
from sys import path

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops
from tqdm import tqdm

import matplotlib.pyplot as plt
from model import HuBMAP_model as Model
from toolfunc import get_data, make_grid, mask2rle

input_size = (512, 512)
with_se = False
modelpaths = [
    "./work_dir/resnest50fpn_se_20e_f0/latest.pth",
    "./work_dir/resnest50fpn_se_20e_f1/latest.pth",
    "./work_dir/resnest50fpn_se_20e_f2/latest.pth",
    "./work_dir/resnest50fpn_se_20e_f3/latest.pth"
]

dataroot = "E:/睡眠分期数据/hubmap/test/"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
window_size = 1024
overlap = window_size//2
mean = np.array([0.485, 0.456, 0.406])*255
std = np.array([0.229, 0.224, 0.225])*255

out_info = {}
models = []
for mp in modelpaths:
    model = Model(with_se=with_se, model_path=None).to(device)
    model.load_state_dict(torch.load(mp)['model'])
    model.eval()
    models.append(model)

filelist = list(filter(lambda fn:fn.endswith(".tiff"), os.listdir(dataroot)))
main_bar = tqdm(filelist)

for fn in main_bar:
    img = get_data(dataroot + fn)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mesh_grid = make_grid(img.shape[:2], window_size, overlap)
    with torch.no_grad():
        sub_bar = tqdm(mesh_grid, leave=False)
        sub_bar.set_description("First epoch pred")
        for (x1, x2, y1, y2) in sub_bar:

            patch = cv2.resize(img[x1:x2, y1:y2], input_size)
            temp = patch.mean((0, 1))
            if sum(abs(temp - temp.mean())) < 30:
                continue
            patch = (patch-mean)/std 
            patch = np.transpose(patch, (2,0,1))
            batch_patch = np.zeros((4, 3, input_size[0], input_size[1]), dtype=patch.dtype)
            batch_patch[0] = patch
            batch_patch[1] = patch[:, ::-1, :]
            batch_patch[2] = patch[:, ::-1, ::-1]
            batch_patch[3] = patch[::-1, :, :]
            batch_patch = torch.FloatTensor(batch_patch).to(device)
            pred_mask = None
            for model in models:
                batch_pred = model(batch_patch).sigmoid().cpu().numpy().astype(np.float16)
                batch_pred[1] = batch_pred[1][:, ::-1, :]
                batch_pred[2] = batch_pred[2][:, ::-1, ::-1]
                batch_pred = batch_pred.mean(0).squeeze()
                if pred_mask is None:
                    pred_mask = batch_pred
                else:
                    pred_mask += pred_mask
            pred_mask = pred_mask/len(models)
            pred_mask = (pred_mask >= 0.5).astype(np.uint8)

            if pred_mask.sum() < 5000:
                continue

            props = regionprops(label(pred_mask, connectivity = pred_mask.ndim))
            for p in props:
                bbox = list(p.bbox)
                if pred_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]].sum() < 5000:
                    pred_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0

            if pred_mask.sum() > 5000:
                pred_mask = cv2.resize(pred_mask, dsize=(window_size, window_size), interpolation = cv2.INTER_NEAREST)
                if mask[x1:x2, y1:y2].shape == pred_mask.shape:
                    mask[x1:x2, y1:y2] = pred_mask

            # if True and pred_mask.sum() > 5000:
            #     sub_bar.write(str(pred_mask.sum()))
            #     plt.figure(figsize=(15,15))
            #     plt.title(str(pred_mask.sum()))
            #     plt.subplot("121")
            #     plt.axis('off')
            #     plt.imshow(img[x1:x2, y1:y2])
            #     plt.imshow(pred_mask,alpha=0.5, cmap='gray')
            #     plt.subplot("122")
            #     plt.axis('off')
            #     plt.imshow(img[x1:x2, y1:y2])
            #     plt.show()

        out_info[len(out_info)] = {'id':fn.split(".")[0], 'predicted': mask2rle(mask)}


submission = pd.DataFrame.from_dict(out_info, orient='index')
if not os.path.exists("./submission"):
    os.makedirs("./submission")
submission.to_csv('./submission/{}_submission.csv'.format("seresnest50_4fold"), index=False)
        