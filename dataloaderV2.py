import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T
from skimage.measure import label, regionprops

import os
from PIL.Image import NONE

import cv2
import pathlib
import numpy as np
import rasterio
from rasterio.windows import Window, transform
import albumentations as A
import pandas as pd
#mean: [0.66406784 0.50002077 0.7019763 ] , std: [0.15964855 0.24647547 0.13597253]
import matplotlib.pyplot as plt
from tqdm import tqdm

def rle_decode(mask_rle, shape=(256, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def make_grid(shape, window=256, min_overlap=32):
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

class HubDataset(D.Dataset):

    def __init__(self, root_dir, transform=None,
                 valid_mode = True,
                 valid_index = [1],
                 imgsize=(512, 512), window=1024, overlap=128, threshold = 200):
        self.path = root_dir
        self.overlap = overlap
        self.window = window
        self.transform = transform
        self.csv = pd.read_csv("{}/train.csv".format(self.path))
        self.threshold = threshold
        self.imgsize = imgsize
        self.valid_mode = valid_mode
        self.valid_index = valid_index
        
        self.build_Transform()
        self.x, self.y = [], []
        self.build_slices()
        self.len = len(self.x)

        
    def build_Transform(self):
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
            # T.Normalize([0.66406784, 0.50002077, 0.7019763],
            #             [0.15964855, 0.24647547, 0.13597253]),
        ])
        self.identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
        self.resizefunc = A.Compose([
            A.Resize(self.imgsize[0], self.imgsize[1])
        ])
        if self.transform is None:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            
                A.OneOf([
                        A.HueSaturationValue(12,12,12, p=0.8),
                        A.CLAHE(clip_limit=2),
                        A.RandomBrightnessContrast(),
                    ], p=0.5),

                A.OneOf([
                    A.RandomContrast(),
                    A.RandomGamma(),
                    A.RandomBrightness(),
                    A.ColorJitter(brightness=0.07, contrast=0.07,
                            saturation=0.1, hue=0.1, always_apply=False, p=0.6),
                    ], p=0.5),

                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ], p=0.3),
                A.ShiftScaleRotate(p=0.5),
            ])
            # self.transform = A.Compose([
            #         A.HorizontalFlip(p=0.5),
            #         A.VerticalFlip(p=0.5),
                    
            #         A.OneOf([
            #             A.RandomContrast(),
            #             A.RandomGamma(),
            #             A.RandomBrightness(),
            #             A.ColorJitter(brightness=0.07, contrast=0.07,
            #                     saturation=0.1, hue=0.1, always_apply=False, p=0.3),
            #             ], p=0.5),

            #         A.OneOf([
            #             A.HueSaturationValue(25,25,25, p=0.6),
            #             A.CLAHE(clip_limit=2),
            #             A.RandomBrightnessContrast(),
            #         ], p=0.5),

            #         A.OneOf([
            #             A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            #             A.GridDistortion(),
            #             A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            #             ], p=0.5),
            #         A.ShiftScaleRotate(p=0.5),
            #     ])
    
    def build_slices(self):
        # self.masks = []
        self.files = []
        self.slices = []
        need_shift = {
            "e79de561c":[5, 10],
            "095bf7a1f":[20, 15],
            "54f2eec69":[-10, -15]
        }
        for i, filename in enumerate(self.csv['id']):

            # filepath = (self.path /(filename+'.tiff')).as_posix()
            filepath = "{}/train/{}.tiff".format(self.path, filename)
            if self.valid_mode:
                if i not in self.valid_index:
                    continue
            else:
                if i in self.valid_index:
                    continue
            # if not os.path.exists(filepath):
            #     continue
            self.files.append(filepath)
            
            print('{}/{}, Transform {}'.format(i+1, len(self.csv['id']), filename))
            with rasterio.open(filepath, transform = self.identity) as dataset:
                mask = rle_decode(self.csv.loc[self.csv['id'] == filename]['encoding'].values[0], dataset.shape)
                if filename in need_shift:
                    print(filename, "需要修正mask")
                    shift_horizontal, shift_vertical = need_shift[filename]
                    h, w = mask.shape
                    mask_temp = np.zeros_like(mask)
                    mask_temp[max(0, -shift_vertical):min(h, h-shift_vertical), max(0, -shift_horizontal):min(w, w-shift_horizontal)] = \
                    mask[max(0, shift_vertical):min(h, h+shift_vertical), max(0, shift_horizontal):min(w, w+shift_horizontal)]
                    mask = mask_temp.copy()
                    del mask_temp
                mask_re = cv2.resize(mask, (1024, 1024))
                props = regionprops(label(mask_re, connectivity = mask_re.ndim))

                # self.masks.append(mask)
                slices = make_grid(dataset.shape, window=self.window, min_overlap=self.overlap)
                # slices = list(slices)
                print("props 处理")
                for p in tqdm(props):
                    bbox = list(p.bbox)
                    bbox[0] = int(bbox[0]* mask.shape[0]/1024)
                    bbox[2] = int(bbox[2]* mask.shape[0]/1024)
                    bbox[1] = int(bbox[1]* mask.shape[1]/1024)
                    bbox[3] = int(bbox[3] * mask.shape[1]/1024)
                    margin = int(((bbox[2]-bbox[0]) + (bbox[3]-bbox[1]))//6)
                    x1,x2,y1,y2 = bbox[0]-margin, bbox[2]+margin, bbox[1]-margin, bbox[3]+margin
                    
                    self.slices.append([i, x1,x2,y1,y2])
                    image = dataset.read([1,2,3],
                            window=Window.from_slices((x1,x2),(y1,y2)))
                    image = np.moveaxis(image, 0, -1)

                    augments = self.resizefunc(image=image, mask=mask[x1:x2,y1:y2])
                    self.x.append(augments['image'])
                    self.y.append(augments['mask'])
                
                print("slices 处理")
                for slc in tqdm(slices):
                    x1,x2,y1,y2 = slc
                    if mask[x1:x2,y1:y2].sum() > self.threshold and (mask[x1:x2,y1:y2].sum()/(abs(x2-x1)*abs(y2-y1))) > 0.015:
                        self.slices.append([i,x1,x2,y1,y2])
                        
                        image = dataset.read([1,2,3],
                            window=Window.from_slices((x1,x2),(y1,y2)))
                        
                        # if image.std().mean() < 10:
                        #     continue
                        
                        # print(image.std().mean(), mask[x1:x2,y1:y2].sum())
                        image = np.moveaxis(image, 0, -1)

                        augments = self.resizefunc(image=image, mask=mask[x1:x2,y1:y2])
                        self.x.append(augments['image'])
                        self.y.append(augments['mask'])
                        # self.x.append(image)
                        # self.y.append(mask[x1:x2,y1:y2])

    def set_transform_flg(self, flg:bool):
        self.valid_mode = flg
    
    # get data operation
    def __getitem__(self, index):
        image, mask = self.x[index], self.y[index]
        # augments = self.resizefunc(image=image, mask=mask)
        # image, mask = augments['image'], augments['mask']
        if self.valid_mode:
            return self.as_tensor(image), mask[None]
        else:
            augments = self.transform(image=image, mask=mask)
            return self.as_tensor(augments['image']), augments['mask'][None]
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

if __name__ == "__main__":
    ds = HubDataset("../data")
    loader = D.DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    for img, mask in loader:
        print(img.shape)
        print(mask.shape)

# id_ = "1e2425f28"
# filepath = "data/{}.tiff".format(id_)
# identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
# train_csv = pd.read_csv('data/train.csv')
# dataset = rasterio.open(filepath, transform = identity)

# mask = rle_decode(train_csv.loc[train_csv['id'] == id_]['encoding'].values[0], dataset.shape)
# mask_re = cv2.resize(mask,(1024, 1024))
# plt.figure(figsize=(5,5))
# plt.axis('off')
# plt.imshow(mask_re)
# plt.show()

# slices = make_grid(dataset.shape, window=1024, min_overlap=256)
# threshold = 100
# for slc in tqdm(slices):
#     x1,x2,y1,y2 = slc
#     if mask[x1:x2,y1:y2].sum() > threshold or np.random.randint(100) > 120:
#         image = dataset.read([1,2,3],
#             window=Window.from_slices((x1,x2),(y1,y2)))
#         image = np.moveaxis(image, 0, -1)
#         trans_img = trfm(image=image, mask=mask[x1:x2,y1:y2])
#         plt.figure(figsize=(10,5))
#         plt.subplot("121")
#         plt.axis('off')
#         plt.imshow(image)
#         plt.imshow(mask[x1:x2,y1:y2], alpha=0.5)

#         plt.subplot("122")
#         plt.axis('off')
#         plt.imshow(trans_img['image'])
#         plt.imshow(trans_img['mask'], alpha=0.5)
#         plt.show()