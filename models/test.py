import mmcv
import torch
import torch.nn.functional as F
from mmcv import Config
from mmseg.models import build_segmentor
cfg = Config.fromfile('upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py')
model = build_segmentor(
        cfg.model)
model = model.backbone

x2 = torch.randn((8,3,512,512))
x2 = model(x2)
lst = []
for i in x2:
    i = F.interpolate(i,size = (64,64), mode='bilinear',align_corners=True)
    lst.append(i)
x2 = torch.cat(lst,dim=1)
print(x2.shape)
