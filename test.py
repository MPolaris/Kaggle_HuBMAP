import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import tifffile
import pandas as pd

from tqdm import tqdm
from model import HuBMAP_model as Model
from Loss import compute_dice
from skimage.measure import label, regionprops

def get_data(id_):
    image = tifffile.imread('{}/train/{}.tiff'.format(dataroot, id_))
    mask_cod  = train_csv.loc[train_csv['id'] == id_]['encoding'].values[0]

    if image.ndim == 5:
        image = image[0,0,:,:,:]
        image = np.transpose(image, (1,2,0))
    
    mask = np.zeros((image.shape[0]*image.shape[1]), dtype=np.uint8)
    
    rle_mask = mask_cod.split()
    positions = map(int, rle_mask[::2])
    lengths = map(int, rle_mask[1::2])
    for pos, le in zip(positions, lengths):
        mask[pos-1:pos+le-1] = 1
   
    mask = mask.reshape((image.shape[1], image.shape[0]))

    return image, mask.T

def make_grid(shape, window=512, min_overlap=128):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2 
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx,ny, 4), dtype=np.int64)
    
    for i in range(nx):
        for j in range(ny):
            slices[i,j] = x1[i], x2[i], y1[j], y2[j]    
    return slices.reshape(nx*ny,4)

root = "./"
model_path = "./work_dir/512_210118resnest50_2/best.pth"
dataroot = "E:/睡眠分期数据/hubmap"
train_csv = pd.read_csv('E:/睡眠分期数据/hubmap/train.csv')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
window_size = 1024
overlap = window_size//2
input_size = (512, 512)
score_threshold = 0.5
mean = np.array([0.485, 0.456, 0.406])*255
std = np.array([0.229, 0.224, 0.225])*255
# mean = np.array([0.66406784, 0.50002077, 0.7019763])*255
# std = np.array([0.15964855, 0.24647547, 0.13597253])*255

out_info = {}

model = Model(with_se=False, model_path=None).to(device)
model.load_state_dict(torch.load(model_path)['model'])
model.eval()
filelist = ["aaa6a05cc"]
main_bar = tqdm(filelist)
for fn in main_bar:
    img, mask = get_data(fn)
    pred_mask = np.zeros(img.shape[:2], dtype=np.float16)
    mesh_grid = make_grid(img.shape[:2], window_size, overlap)
    with torch.no_grad():
        sub_bar = tqdm(mesh_grid, leave=False)
        sub_bar.set_description("First epoch pred")
        for (x1, x2, y1, y2) in sub_bar:
            patch = cv2.resize(img[x1:x2, y1:y2], input_size)
            temp = patch.mean((0, 1))
            if sum(abs(temp - temp.mean())) < 12:
                continue
            patch = (patch-mean)/std
            patch = np.transpose(patch, (2,0,1))
            img_slice = torch.FloatTensor(patch[np.newaxis,:,:,:]).to(device)
            pred = model(img_slice)

            # pred2 = model(torch.flip(img_slice, [0, 3]))
            # pred2 = torch.flip(pred2, [3, 0])

            # pred3 = model(torch.flip(img_slice, [1, 2]))
            # pred3 = torch.flip(pred3, [2, 1])

            # pred = (pred1 + pred2 + pred3)/3.0
            pred = F.interpolate(pred, size=(window_size, window_size), mode="nearest")
            pred = torch.sigmoid(pred).cpu().numpy()[0, 0].astype(np.float16)
            if pred_mask[x1:x2, y1:y2].shape == pred.shape:
                if x1>overlap and y1>overlap:
                    pred_mask[x1:x1+overlap, y1:y2] = (pred_mask[x1:x1+overlap, y1:y2] + pred[:overlap, :])/2
                    pred_mask[x1+overlap:x2, y1:y1+overlap] = (pred_mask[x1+overlap:x2, y1:y1+overlap] + pred[overlap:, :overlap])/2
                    pred_mask[x1+overlap:x2, y1+overlap:y2] = pred[overlap:, overlap:]
                else:
                    pred_mask[x1:x2, y1:y2] = pred
        sub_bar.clear()
        sub_bar.close()
        for score_threshold in [0.4, 0.6, 0.5]:
            pred_mask_first = (pred_mask >= score_threshold).astype(np.uint8)
            dice_1 = compute_dice(pred_mask_first, mask)
            main_bar.write("{} score_threshold:{},dice1:{:^7.4f}".format(fn, score_threshold, dice_1))

        pred_mask_re = cv2.resize(pred_mask_first, (1024, 1024))
        del mesh_grid
        del pred_mask_first
        props = regionprops(label(pred_mask_re, connectivity = pred_mask_re.ndim))
        del pred_mask_re
        sub_bar = tqdm(props, leave=False)
        sub_bar.set_description("second epoch pred")
        random_scalors = [6, 10]
        pred_mask_arr = np.zeros((len(random_scalors), img.shape[0], img.shape[1]), dtype=np.float16)
        for p in sub_bar:
            bbox = list(p.bbox)
            bbox[0] = int(bbox[0]* img.shape[0]/1024)
            bbox[2] = int(bbox[2]* img.shape[0]/1024)
            bbox[1] = int(bbox[1]* img.shape[1]/1024)
            bbox[3] = int(bbox[3] * img.shape[1]/1024)

            if bbox[2] - bbox[0] < 5 and bbox[3] - bbox[1] < 5:
                continue

            for i in range(len(random_scalors)):
                margin = int(((bbox[2]-bbox[0]) + (bbox[3]-bbox[1]))//random_scalors[i])
                patch = cv2.resize(img[max(bbox[0]-margin, 0):bbox[2]+margin, max(bbox[1]-margin, 0):bbox[3]+margin], input_size)
                patch = (patch-mean)/std
                patch = np.transpose(patch, (2,0,1))
                img_slice = torch.FloatTensor(patch[np.newaxis,:,:,:]).to(device)
                pred = model(img_slice)

                # pred2 = model(torch.flip(img_slice, [0, 3]))
                # pred2 = torch.flip(pred2, [3, 0])

                # pred3 = model(torch.flip(img_slice, [1, 2]))
                # pred3 = torch.flip(pred3, [2, 1])

                # pred = (pred1 + pred2 + pred3)/3.0

                pred = F.interpolate(pred, size=(bbox[2]+margin - max(bbox[0]-margin, 0), bbox[3]+margin - max(bbox[1]-margin, 0)), mode="nearest")
                pred = torch.sigmoid(pred).cpu().numpy()[0, 0].astype(np.float16)
                if pred_mask_arr[i, max(bbox[0]-margin, 0):bbox[2]+margin, max(bbox[1]-margin, 0):bbox[3]+margin].shape == pred.shape:
                    pred_mask_arr[i, max(bbox[0]-margin, 0):bbox[2]+margin, max(bbox[1]-margin, 0):bbox[3]+margin] = pred

        pred_mask_arr = np.mean(pred_mask_arr, axis=0)
        for score_threshold in [0.5]:
            for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                _pred_mask = a*pred_mask_arr + (1-a)*pred_mask
                _pred_mask = (_pred_mask >= score_threshold).astype(np.uint8)
                dice_2 = compute_dice(_pred_mask, mask)
                main_bar.write("{} st:{} a:{}, dice2:{:^7.4f}".format(fn, score_threshold, a, dice_2))
                del _pred_mask
        del pred_mask_arr