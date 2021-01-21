import torch
import torch.nn as nn

def compute_dice(preds, targets):
    return (2*(preds*targets).sum() + 1e-5)/(preds.sum() + targets.sum() + 1e-5)

class DiceLoss(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
 
    def forward(self, preds, targets):
        if self.use_sigmoid:
            preds = torch.sigmoid(preds)
        score = 1 - compute_dice(preds, targets)
        return score

class SoftDiceLoss(nn.Module):
    def __init__(self, use_sigmoid=True, smooth=1e-5, dims=(-2,-1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
        self.use_sigmoid = use_sigmoid
    
    def forward(self, preds, targets):
        if self.use_sigmoid:
            preds = torch.sigmoid(preds)

        tp = (preds * targets).sum(self.dims)
        fp = (preds * (1 - targets)).sum(self.dims)
        fn = ((1 - preds) * targets).sum(self.dims)
        
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

class FocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, gamma=2):
        super(FocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma

    def forward(self, preds, targets):
        if self.use_sigmoid:
            preds = torch.sigmoid(preds)

        loss = torch.pow(1-preds, self.gamma)*targets*torch.log(preds + 1e-5) + torch.pow(preds, self.gamma)*(1-targets)*torch.log(1-preds + 1e-5)
        return -loss.mean()

if __name__ =="__main__":
    loss1 = FocalLoss(gamma = 1.5)
    loss2 = nn.BCEWithLogitsLoss()
    pred = torch.randn(2,1,64,64)
    target = torch.randint(0, 2, (2,64,64)).float()
    target.unsqueeze_(1)
    # target = torch.zeros_like(pred)
    print(loss1(pred, target), loss2(pred, target))