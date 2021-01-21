# import sys
# sys.path.append("F:/DDSM/codeForObjectDetection/")
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
import torch.nn.functional as F

import matplotlib.pyplot as plt
import argparse
import importlib
import numpy as np
import random
from tqdm import tqdm

from Loss import DiceLoss, compute_dice, FocalLoss, SoftDiceLoss
# from dataloader import Dataloader
from dataloaderV2 import HubDataset
from model import HuBMAP_model as Model

import warnings
warnings.filterwarnings('ignore')

class Train_Config:
    def __init__(self) -> None:
        super(Train_Config, self).__init__()
        self.config_name = "test"
        self.workname = "./work_dir/test"
        self.train_root = "E:/睡眠分期数据/hubmap"
        self.batch_size = 32
        self.max_epoch = 12
        self.lr_config = dict(warmup='linear',warmup_iters=500, step=[8, 11])
        self.optimizer = dict(type='SGD', lr=self.batch_size*0.00125, momentum=0.9, weight_decay=0.0001)
        # self.optimizer = dict(type='Adam', lr=0.001)
        self.with_se = True
        self.seed = 26
        self.img_size = (512, 512)
        self.load_from = "./model_state/resnest50_fpn_coco.pth"
        self.resume = None
        self.use_GPU = True
        self.valid_index = [1]

def set_seeds(seed=26):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

parser = argparse.ArgumentParser() 
parser.add_argument('-filepath', type=str, help='input config file path')
args = parser.parse_args()
config_path = args.filepath.split("\\")
config_path = ".".join(config_path)
config = importlib.import_module(config_path)

# config = Train_Config()
set_seeds(getattr(config, "seed", 26))

#初始化工作目录
device = torch.device('cuda') if torch.cuda.is_available() and getattr(config, "use_GPU", True) else torch.device('cpu')
workname = getattr(config,'workname','./work_dir/defaultworkname')
if not os.path.exists(workname):
    os.makedirs(workname)

#初始化模型
# print(getattr(config, 'with_se', False))
model = Model(with_se=getattr(config, 'with_se', False),model_path=getattr(config, 'load_from', None))
model.to(device)

#初始化dataloder
trainDL = D.DataLoader(HubDataset("E:/睡眠分期数据/hubmap", imgsize=getattr(config, 'img_size', (256, 256)), valid_mode=False, valid_index=getattr(config, "valid_index", [1]))
    , batch_size=getattr(config, 'batch_size', 16), shuffle=True, num_workers=0)

valDL = D.DataLoader(HubDataset("E:/睡眠分期数据/hubmap", imgsize=getattr(config, 'img_size', (256, 256)), valid_mode=True, valid_index=getattr(config, "valid_index", [1]))
    , batch_size=getattr(config, 'batch_size', 16), shuffle=False, num_workers=0)

#初始化优化器
optimizer_config = getattr(config, "optimizer")
optimizer = getattr(optim, optimizer_config["type"])
del optimizer_config["type"]
optimizer_config['params'] = model.parameters()
optimizer = optimizer(**optimizer_config)

#初始化loss
# focal_loss_fn = FocalLoss()
bce_loss_fn = nn.BCEWithLogitsLoss()
dice_loss_fn = DiceLoss()
# dice_loss_fn = SoftDiceLoss()

#初始化训练配置
batchsize = config.batch_size
max_epoch = config.max_epoch
lr_config = config.lr_config

lossForTrain = []
diceForTrain = []
lossForVal = []
diceForVal = []

# print(config.optimizer)

main_bar = tqdm(range(1,max_epoch+1))
header = r'''
----------------------------------------
|      |     Train     |   Validation  |
|Epoch |  Loss |  dice | Loss |  dice  |
----------------------------------------'''
raw_line = '\u2502{:^6d}' + '\u2502{:^7.4f}'*2 + '\u2502{:^7.4f}'*2 + '\u2502'
main_bar.write(header)
with open(workname+'/train_log.txt', "w") as fp:
    fp.write(header + "\n")


for epoch in main_bar:
    #train
    model.train()
    # ds.set_transform_flg(True)
    trainloss = []
    traindice = []
    train_bar = tqdm(trainDL, leave=False)
    for batch_data, batch_label in train_bar:
        train_bar.set_description("Training")
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        batch_label = batch_label.float().to(device)
        pred = model(batch_data)
        
        bceloss = bce_loss_fn(pred, batch_label)
        # focalloss = focal_loss_fn(pred, batch_label)
        diceloss = dice_loss_fn(pred, batch_label)
        loss = 0.5*bceloss + 0.5*diceloss
        # loss = 0.5*focalloss + 0.5*diceloss
        
        loss.backward()
        optimizer.step()

        trainloss.append(loss.item())
        traindice.append(1-diceloss.item())
        train_bar.set_postfix(Loss=trainloss[-1], traindice=traindice[-1])
    train_bar.close()
    trainloss = np.mean(trainloss)
    traindice = np.mean(traindice)
    lossForTrain.append(trainloss)
    diceForTrain.append(traindice)
    #validation
    model.eval()
    # ds.set_transform_flg(False)
    valloss = []
    valdice = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        val_bar = tqdm(valDL, leave=False)
        for batch_data,batch_label in val_bar:
            val_bar.set_description("Testing")
            batch_data = batch_data.to(device)
            batch_label = batch_label.float().to(device)
            pred = model(batch_data)
            bceloss = bce_loss_fn(pred, batch_label)
            # focalloss = focal_loss_fn(pred, batch_label)
            diceloss = dice_loss_fn(pred, batch_label)
            loss = 0.5*bceloss + 0.5*diceloss
            # loss = 0.5*focalloss + 0.5*diceloss
            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            pred = torch.sigmoid(pred)
            pred = torch.where(pred > 0.5, one, zero)
            dice = compute_dice(pred, batch_label)

            valloss.append(loss.item())
            valdice.append(dice.item())
            val_bar.set_postfix(Loss=valloss[-1], valdice=valdice[-1])

    valloss = np.mean(valloss)
    valdice = np.mean(valdice)
    lossForVal.append(valloss)
    diceForVal.append(valdice)

    main_bar.write(raw_line.format(epoch, trainloss, traindice, valloss, valdice))
    main_bar.write("----------------------------------------")
    with open(workname+'/train_log.txt', "a+") as fp:
        fp.write(raw_line.format(epoch, trainloss, traindice, valloss, valdice) + "\n")
        fp.write("----------------------------------------\n")

    #调整学习速率
    if epoch in lr_config['step']:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*0.1
    
    #保存模型
    checkpoint = {
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "epoch":epoch,
        "Info":[
            lossForTrain,
            diceForTrain,
            lossForVal,
            diceForVal
        ]
        }
    torch.save(checkpoint,workname + "/latest.pth")
    if epoch > 1 and lossForVal[-1] == min(lossForVal):
        torch.save(checkpoint, workname + "/best.pth")
    
    if epoch > 2:
        plt.figure(figsize=(50, 20))
        plt.subplot('121')
        plt.title("Loss")
        plt.plot(lossForTrain,label='TrainLoss')
        plt.plot(lossForVal,label='ValLoss')
        plt.legend()

        plt.subplot('122')
        plt.title("Dice")
        plt.plot(diceForTrain,label='TrainDice')
        plt.plot(diceForVal,label='ValDice')
        plt.legend()

        plt.savefig(workname + '/train_log.jpg')
        plt.close()

print("训练完成了")

