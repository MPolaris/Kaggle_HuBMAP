config_name = "resnest50fpn_se_20e_specifymeanstd"
workname = "./work_dir/resnest50fpn_se_20e_specifymeanstd"
train_root = "E:/HuBMAP_data/"
with_se = True
batch_size = 16
max_epoch = 20
lr_config = dict(warmup='linear',warmup_iters=500, step=[10, 17])
optimizer = dict(type='SGD', lr=0.00125*batch_size, momentum=0.9, weight_decay=0.0001)
seed = 26
img_size = (512, 512)
valid_index = [1]
# T.Normalize([0.66406784, 0.50002077, 0.7019763],
#             [0.15964855, 0.24647547, 0.13597253]),

load_from = "./model_state/resnest50_fpn_coco.pth"
resume = None
use_GPU = True

