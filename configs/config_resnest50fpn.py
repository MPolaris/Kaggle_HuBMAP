config_name = "resnest50fpn"
workname = "./work_dir/resnest50fpn"
train_root = "E:/HuBMAP_data/"
batch_size = 16
max_epoch = 12
lr_config = dict(warmup='linear',warmup_iters=500, step=[8, 11])
optimizer = dict(type='SGD', lr=0.00125*batch_size, momentum=0.9, weight_decay=0.0001)
seed = 26
img_size = (512, 512)
valid_index = [1]


load_from = "./old_code/resnest50_fpn_coco.pth"
resume = None
use_GPU = True

