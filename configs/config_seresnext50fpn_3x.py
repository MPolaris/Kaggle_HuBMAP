config_name = "resnest50fpn_3x"
workname = "./work_dir/resnest50fpn_3x"
train_root = "E:/HuBMAP_data/"
backbone_name = "se_resnext"
batch_size = 30
max_epoch = 30
lr_config = dict(warmup='linear',warmup_iters=500, step=[18, 26])
optimizer = dict(type='SGD', lr=min(0.00125*batch_size, 0.02), momentum=0.9, weight_decay=0.0001)
seed = 26
img_size = (512, 512)
valid_index = [1]


load_from = ["./model_state/se_resnext50_32x4d.pth"]
resume = None
use_GPU = True

