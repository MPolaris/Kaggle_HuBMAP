import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import tifffile
import pandas as pd
import gc

import argparse
import importlib

from tqdm import tqdm
from model import HuBMAP_model as Model
from skimage.measure import label, regionprops

def get_data(path):
    image = tifffile.imread(path)
    if image.ndim == 5:
        image = image[0,0,:,:,:]
        image = np.transpose(image, (1,2,0))
    return image

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

def mask2rle(mask):
    ''' takes a 2d boolean numpy array and turns it into a space-delimited RLE string '''
    mask = mask.T.reshape(-1) # make 1D, column-first
    mask = np.pad(mask, 1, mode="constant") # make sure that the 1d mask starts and ends with a 0
    starts = np.nonzero((~mask[:-1]) & mask[1:])[0] # start points
    ends = np.nonzero(mask[:-1] & (~mask[1:]))[0] # end points
    rle = np.empty(2 * starts.size, dtype=int) # interlacing...
    rle[0::2] = starts + 1# ...starts...
    rle[1::2] = ends - starts # ...and lengths
    rle = ' '.join([ str(elem) for elem in rle ]) # turn into space-separated string
    return rle

parser = argparse.ArgumentParser() 
parser.add_argument('-filepath', type=str, help='input config file path')
args = parser.parse_args()
config_path = args.filepath.split("\\")
config_path = ".".join(config_path)
config = importlib.import_module(config_path)
config_name = getattr(config, 'config_name', 'None')
workname = getattr(config, 'workname', './work_dir/defaultworkname')

root = "./"
model_path = "{}/latest.pth".format(workname)
dataroot = "E:/睡眠分期数据/hubmap/test/"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
window_size = 1024
overlap = window_size//2
input_size = (512, 512)
mean = np.array([0.485, 0.456, 0.406])*255
std = np.array([0.229, 0.224, 0.225])*255
# mean = np.array([0.66406784, 0.50002077, 0.7019763])*255
# std = np.array([0.15964855, 0.24647547, 0.13597253])*255

out_info = {}

model = Model(with_se=getattr(config, 'with_se', False), model_path=None).to(device)
model.load_state_dict(torch.load(model_path)['model'])
model.eval()
filelist = list(filter(lambda fn:fn.endswith(".tiff"), os.listdir(dataroot)))
filelist = filelist
main_bar = tqdm(filelist)
for fn in main_bar:
    img = get_data(dataroot + fn)
    pred_mask = np.zeros(img.shape[:2], dtype=np.float16)
    mesh_grid = make_grid(img.shape[:2], window_size, overlap)
    with torch.no_grad():
        score_threshold = 0.5
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
            # pred1 = model(img_slice)

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
        pred_mask_first = (pred_mask >= score_threshold).astype(np.uint8)
        pred_mask_re = cv2.resize(pred_mask_first, (1024, 1024))
        del mesh_grid
        del pred
        del pred_mask_first
        props = regionprops(label(pred_mask_re, connectivity = pred_mask_re.ndim))
        del pred_mask_re
        gc.collect()
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

            for i in range(len(random_scalors)):
                margin = int(((bbox[2]-bbox[0]) + (bbox[3]-bbox[1]))//random_scalors[i])
                patch = cv2.resize(img[max(bbox[0]-margin, 0):bbox[2]+margin, max(bbox[1]-margin, 0):bbox[3]+margin], input_size)
                patch = (patch-mean)/std
                patch = np.transpose(patch, (2,0,1))
                img_slice = torch.FloatTensor(patch[np.newaxis,:,:,:]).to(device)
                pred = model(img_slice)
                pred = F.interpolate(pred, size=(bbox[2]+margin - max(bbox[0]-margin, 0), bbox[3]+margin - max(bbox[1]-margin, 0)), mode="nearest")
                pred = torch.sigmoid(pred).cpu().numpy()[0, 0].astype(np.float16)
                if pred_mask_arr[i, max(bbox[0]-margin, 0):bbox[2]+margin, max(bbox[1]-margin, 0):bbox[3]+margin].shape == pred.shape:
                    pred_mask_arr[i, max(bbox[0]-margin, 0):bbox[2]+margin, max(bbox[1]-margin, 0):bbox[3]+margin] = pred

        score_threshold = 0.5
        a = 0.4
        pred_mask_arr = np.mean(pred_mask_arr, axis=0)
        pred_mask = a*pred_mask_arr + (1-a)*pred_mask
        pred_mask = (pred_mask >= score_threshold).astype(np.uint8)
        del pred_mask_arr
    
    del img
    gc.collect()
        
    out_info[len(out_info)] = {'id':fn.split(".")[0], 'predicted': mask2rle(pred_mask)}

submission = pd.DataFrame.from_dict(out_info, orient='index')
if not os.path.exists("./submission"):
    os.makedirs("./submission")
submission.to_csv('./submission/{}_submission.csv'.format(config_name), index=False)