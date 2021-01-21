import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .fpn import FPN
from .resnet import resnest50
from .se_resnext import se_resnext50_32x4d

class HuBMAP_model(nn.Module):
    def __init__(self, num_class=1, use_sigmoid=True, with_se=False, model_path="./resnest50_fpn_coco.pth"):
        super(HuBMAP_model, self).__init__()
        self.backbone = resnest50(with_se=with_se)
        self.fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=4)
        num_class = num_class if use_sigmoid else num_class+1
        self.predictor = nn.Conv2d(256, num_class, 3, 1, 1, bias=True)

        self.init_weights()
        if isinstance(model_path, str):
            #加载整个模型
            self.load_state_dict(torch.load(model_path), strict=False)
        elif isinstance(model_path, list):
            #只加载backbone
            self.backbone.load_state_dict(torch.load(model_path[0]), strict=False)
            # self.fpn.load_state_dict(torch.load(model_path[1]))
            
    def init_weights(self):
        # self.backbone.init_weights()
        self.fpn.init_weights()
        import math
        n = self.predictor.kernel_size[0] * self.predictor.kernel_size[1] * self.predictor.out_channels
        self.predictor.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.backbone(x)
        # for i in range(len(x)):
        #     x[i] = F.interpolate(x[i], scale_factor=2, mode="nearest")
        # x = cp.checkpoint(self.fpn, x)
        x = self.fpn(x)
        # pred = cp.checkpoint(self.predictor, x)
        pred = self.predictor(x)
        pred = F.interpolate(pred, size=input_size, mode="nearest")
        return pred

if __name__ == "__main__":
    with torch.no_grad():
        img = torch.randn(1,3,512,512)
        model = HuBMAP_model(model_path="./resnest50_fpn_coco.pth")
        model.eval()
        pred = model(img)
        #torch.save(model.state_dict(), "resnest50_fpn_coco.pth")
        print(pred.shape)