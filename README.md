<div align="center"> <h1>HuBMAP比赛代码</h1> </div>
None
## 配置文件config
参照配置文件夹[configs](./configs)

## 运行方法
单个训练: python .\train_args.py -filepath configs\config_resnest50fpn_se_20e
<br/>
批量训练：windowns编写run.bat, liunx编写run.sh
<br/>
本地测试：运行test.py，手动修改相应的modelpath、以及model的配置。
<br/>
生成提交：python .\train_args.py -filepath configs\config_resnest50fpn_se_20e

## 模型：
Supported backbones:
- [ ] ResNet
- [x] ResNeXt
- [x] ResNeSt

Supported attention modual:
- [x] SE
- [ ] CBAM

## 损失函数：
- [x] FocalLoss
- [x] BCE
- [x] Dice

## TODO:
- [X] 支持可以在config里面配置主干网络
- [ ] 继续优化整个训练流程
