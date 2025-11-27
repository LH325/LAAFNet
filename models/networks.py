import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
# from .starnet import StarNet
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_

import math

import torch.nn.functional as F

# Coordinate Attention协调注意力机制
import torch
from torch import nn
# from mmcv.cnn import ConvModule
import numpy as np
from einops import rearrange
from .DCnV2 import DeformConv2d

from .starnet import StarNet

from torch.fft import fft2, fftshift, ifft2, ifftshift
import torch
import torch.fft


def extract_high_frequency(image, radius_ratio=0.1):
    """
    提取图像的高频部分

    参数:
        image: 输入图像, 形状 [B, W, H]
        radius_ratio: 保留高频的比例，越小保留的频率越高

    返回:
        高频部分图像，形状 [B, W, H]
    """
    B, W, H = image.shape

    # 1. 傅里叶变换
    fft = torch.fft.fft2(image)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    # 2. 创建掩码去除低频
    mask = torch.ones(B, W, H, dtype=torch.float32, device=image.device)
    center_w, center_h = W // 2, H // 2
    radius_w, radius_h = int(W * radius_ratio), int(H * radius_ratio)

    mask[:, center_w - radius_w:center_w + radius_w,
    center_h - radius_h:center_h + radius_h] = 0

    # 应用掩码
    fft_shifted_high = fft_shifted * mask

    # 3. 逆变换回空间域
    fft_high = torch.fft.ifftshift(fft_shifted_high, dim=(-2, -1))
    image_high = torch.fft.ifft2(fft_high).real

    return image_high


# 使用示例
# 假设你的图像是 batch_size=4, 宽度=256, 高度=256
# image = torch.randn(4, 256, 256)
# high_freq = extract_high_frequency(image, radius_ratio=0.1)
class RCA(nn.Module):
    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, dw_size=(1, 1), padding=(0, 0), stride=1,
                 square_kernel_size=2, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def sge(self, x):
        # [N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w  # .repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather)  # [N, 1, C, 1]

        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc

        return out


class Denosing(torch.nn.Module):
    def __init__(self, in_channel, ratio):
        super(Denosing, self).__init__()
        self.avepool = torch.nn.AdaptiveAvgPool2d(1)

        self.linear1 = torch.nn.Linear(in_channel, in_channel // ratio)
        self.linear2 = torch.nn.Linear(in_channel // ratio, in_channel)

        self.sigmoid = torch.nn.Sigmoid()
        self.Relu = torch.nn.SiLU()

    def forward(self, input):
        b, c, w, h = input.shape
        x = self.avepool(input)
        x = x.view([b, c])
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = x.view([b, c, 1, 1])

        return input * x


class TransformerCBAMLayer(nn.Module):
    def __init__(self, channel1, channel2, reduction=16, spatial_kernel=7):
        super(TransformerCBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool_w_1 = nn.AdaptiveMaxPool2d((1, None))
        self.max_pool_h_1 = nn.AdaptiveMaxPool2d((None, 1))
        self.avg_pool_w_1 = nn.AdaptiveAvgPool2d((1, None))
        self.avg_pool_h_1 = nn.AdaptiveAvgPool2d((None, 1))

        self.max_pool_w_2 = nn.AdaptiveMaxPool2d((1, None))
        self.max_pool_h_2 = nn.AdaptiveMaxPool2d((None, 1))
        self.avg_pool_w_2 = nn.AdaptiveAvgPool2d((1, None))
        self.avg_pool_h_2 = nn.AdaptiveAvgPool2d((None, 1))
        # shared MLP

        self.mlp1 = nn.Sequential(
            nn.Conv2d(channel1, channel1 // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            nn.Conv2d(channel1 // reduction, channel1, 1, bias=False)
        )
        # spatial attention

        self.mlp2 = nn.Sequential(
            nn.Conv2d(channel2, channel2 // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel2 // reduction, channel2, 1, bias=False)
        )
        # spatial attention
        self.cbam_conv = nn.Conv2d(4, 1, kernel_size=spatial_kernel,
                                   padding=spatial_kernel // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        if x1.shape[1] == 256:
            x1 = F.interpolate(x1, size=(128, 128), mode='bilinear', align_corners=True)

        max_out1 = self.mlp1(self.max_pool_w_1(x1) + self.max_pool_h_1(x1))
        avg_out1 = self.mlp1(self.avg_pool_w_1(x1) + self.avg_pool_h_1(x1))
        max_out2 = self.mlp2(self.max_pool_w_2(x2) + self.max_pool_h_2(x2))
        avg_out2 = self.mlp2(self.avg_pool_w_2(x2) + self.avg_pool_h_2(x2))

        channel_out = self.sigmoid(max_out1 + avg_out1)
        x1 = channel_out * x1
        max_out1, _ = torch.max(x1, dim=1, keepdim=True)
        # print('max_out:',max_out.shape)
        avg_out1 = torch.mean(x1, dim=1, keepdim=True)
        min_out1 = torch.min(x1, dim=1, keepdim=True)

        channel_out2 = self.sigmoid(max_out2 + avg_out2)
        x2 = channel_out2 * x2
        max_out2, _ = torch.max(x2, dim=1, keepdim=True)
        # print('max_out:',max_out.shape)
        avg_out2 = torch.mean(x2, dim=1, keepdim=True)


        spatial_out = self.sigmoid(self.cbam_conv(torch.cat([max_out1, avg_out1, max_out2, avg_out2], dim=1)))

        x = torch.cat([x1, x2], dim=1)

        x = spatial_out * x

        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        # self.pa1 = nn.Conv2d(dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)

        self.linear1 = torch.nn.Linear(dim, dim // 16)
        self.linear2 = torch.nn.Linear(dim // 16, dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.Relu = torch.nn.SiLU()

    def forward(self, pattn1):
        b, c, _, _ = pattn1.shape
        pattn1 = pattn1.view([b, c])
        pattn1 = self.linear1(pattn1)
        pattn1 = self.Relu(pattn1)
        pattn1 = self.linear2(pattn1)
        pattn1 = self.sigmoid(pattn1)
        pattn1 = pattn1.view([b, c, 1, 1])
        return pattn1


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = torch.cat([x, y], dim=1)

        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * initial + (1 - pattn2) * y
        result = self.conv(result)
        return result


class LAAFNet(nn.Module):
    def __init__(self, channel1, channel2, reduction=16, spatial_kernel=7):
        super(LAAFNet, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp1 = nn.Sequential(

            nn.Conv2d(channel1, channel1 // reduction, 1, bias=False),

            nn.ReLU(inplace=True),

            nn.Conv2d(channel1 // reduction, channel1, 1, bias=False)
        )

        self.mlp2 = nn.Sequential(

            nn.Conv2d(channel2, channel2 // reduction, 1, bias=False),

            nn.ReLU(inplace=True),

            nn.Conv2d(channel2 // reduction, channel2, 1, bias=False)
        )

        self.pixel_conv = nn.Conv2d(4, 4, kernel_size=spatial_kernel,
                                    padding=spatial_kernel // 2, bias=False)
        self.pixel_weight_conv = nn.Conv2d(4, 1, kernel_size=spatial_kernel,
                                   padding=spatial_kernel // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        if x1.shape[1] == 256:
            x1 = F.interpolate(x1, size=(128, 128), mode='bilinear', align_corners=True)

        max_out1 = self.mlp1(self.max_pool(x1))
        avg_out1 = self.mlp1(self.avg_pool(x1))
        max_out2 = self.mlp2(self.max_pool(x2))
        avg_out2 = self.mlp2(self.avg_pool(x2))
        channel_out = self.sigmoid(max_out1 + avg_out1)

        x1 = channel_out * x1
        channel_out2 = self.sigmoid(max_out2 + avg_out2)

        x2 = channel_out2 * x2

        max_out1, _ = torch.max(x1, dim=1, keepdim=True)
        avg_out1 = torch.mean(x1, dim=1, keepdim=True)
        max_out2, _ = torch.max(x2, dim=1, keepdim=True)
        avg_out2 = torch.mean(x2, dim=1, keepdim=True)

        spatial_out = self.sigmoid(
            self.pixel_weight_conv(self.pixel_conv(torch.cat([max_out1, avg_out1, max_out2, avg_out2], dim=1))))

        x = torch.cat([x1, x2], dim=1)

        x = x * spatial_out

        return x

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)
    elif args.net_G == 'LAAFNet':
        net = Base_EncoderDecoder(input_nc=3, output_nc=2)
    elif args.net_G == 'starnet':
        net = StarNets4(input_nc=3, output_nc=2)
    elif args.net_G == 'mobilenetV2':
        net = mobilenet(input_nc=3, output_nc=2)
    elif args.net_G == 'VIT':
        net = VIT(input_nc=3, output_nc=2)

    # elif args.net_G == 'swint':
    #     net = swintransformer(input_nc=3,output_nc=2)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=False,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=False,
                                          replace_stride_with_dilation=[False, True, True])
            expand = 4

        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError

        '''
        Starnet VIT需要
        '''
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)



    def forward(self, x1, x2):

        x4, x8, x1 = self.forward_single(x1)

        x42, x82, x2 = self.forward_single2(x2)

        x4 = self.se1(x4)
        x42 = self.se1(x42)

        x1 = self.aspp(x1)
        x2 = self.aspp(x2)

        xl = self.cbamlow(x4,x42)

        xm = self.cbammid(x8,x82)
        xd = self.cbam(x1,x2)

        xm = F.interpolate(xm, size=(xl.size(2), xl.size(3)), mode='bilinear',
                          align_corners=True)

        x = self.cat_conv(torch.cat([xl, xm, xd], dim=1))
        out1 = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
        out1 = self.cls_conv(out1)

        return out1



    def forward_single(self, x):

        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64

        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512

        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)

        return x_4, x_8, x

    def forward_single2(self, x):

        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64

        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512

        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)

        return x_4, x_8, x

from .xception import xception
from .deeplabv3_plus import ASPP, MobileNetV2
from .mobilenetv2 import mobilenetv2


# from .swin_transformer import swin_t, Swin_T_Weights
# from vmamba_transfer import test
# class swintransformer(ResNet):
#     def __init__(self,input_nc, output_nc):
#         super(swintransformer,self).__init__(input_nc,output_nc)
#         self.backbone1 = swin_t(weights=Swin_T_Weights.DEFAULT).features
#         self.backbone2 = swin_t(weights=Swin_T_Weights.DEFAULT).features
#         self.conv2d = nn.Conv2d(768,128,1,1,1)
#         self.batch = nn.BatchNorm2d(128)
#         self.relu = nn.ReLU()

#         self.conv2d2 = nn.Conv2d(128, 2, 1, 1, 1)
#         self.batch2 = nn.BatchNorm2d(2)
#         self.relu2 = nn.ReLU()
#     def forward(self, x1, x2):
#         H, W = x1.size(2), x1.size(3)
#         x1 = self.backbone1(x1)
#         x2 = self.backbone2(x2)
#         x1 = x1.permute(0,3,1,2)
#         x2 = x2.permute(0,3,1,2)
#         x = x1*x2
#         x = self.conv2d(x)
#         x = self.batch(x)
#         x = self.relu(x)
#         x = F.interpolate(x, size=(64, 64), mode='bilinear',
#                           align_corners=True)

#         x = self.conv2d2(x)
#         x = self.batch2(x)
#         x = self.relu2(x)
#         x = F.interpolate(x, size=(512, 512), mode='bilinear',
#                           align_corners=True)
#         return x
'''
LAAFNet (完整实现)
'''
class Base_EncoderDecoder(ResNet):
    def __init__(self, input_nc, output_nc, downsample_factor=8, in_channels=320, low_level_channels=24, num_classes=2,
                 pretrained=False):
        super(Base_EncoderDecoder, self).__init__(input_nc, output_nc)
        self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        self.aspp2 = ASPP(dim_in=32, dim_out=32, rate=16 // downsample_factor)

        self.se1 = Denosing(48, 16)
        self.se2 = Denosing(64, 16)

        self.la1 = LAAFNet(48, 64)
        self.la2 = LAAFNet(64, 512)

        self.la3 = LAAFNet(256, 32)


        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(

            nn.Conv2d(48 + 64 + 256 + 64 + 512 + 32, 256, 3, stride=1, padding=1),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )


        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)


    def _forward_transformer(self, x):


        x += self.pos_embedding
        x = self.transformer(x)

        return x
    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, 2, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape

        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def forward(self, x1, x2):

        H, W = x1.size(2), x1.size(3)

        low_level_features, middle_features, x1 = self.backbone(x1)
        x1 = self.aspp(x1)
        low_level_features = self.shortcut_conv(low_level_features)
        low_level_features = self.se1(low_level_features)

        x_4, x_8, x2 = self.forward_single(x2)

        co_low1 = low_level_features.mean(dim=1)
        co_low2 = x_4.mean(dim=1)
        co_mid1 = middle_features.mean(dim=1)
        co_mid2 = x_8.mean(dim=1)
        co_deep1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)), mode='bilinear',
                          align_corners=True)
        co_deep1 = co_deep1.mean(dim=1)
        co_deep2 = x2.mean(dim=1)
        K = [co_low1,co_low2,co_mid1,co_mid2,co_deep1,co_deep2]
        K = [extract_high_frequency(i) for i in K]

        x_4 = self.se2(x_4)

        x2 = self.aspp2(x2)

        low_level_features = self.la1(low_level_features, x_4)

        middle_features = self.la2(middle_features, x_8)

        x = self.la3(x1, x2)

        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        middle_features = F.interpolate(middle_features, size=(low_level_features.size(2), low_level_features.size(3)),
                                        mode='bilinear',
                                        align_corners=True)


        x = self.cat_conv(torch.cat([x, low_level_features, middle_features], dim=1))
        out1 = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        out1 = self.cls_conv(out1)

        K=4
        return out1, K

class mobilenet(ResNet):
    def __init__(self, input_nc, output_nc, downsample_factor=8, in_channels=320, low_level_channels=24, num_classes=2,
                 pretrained=False):
        super(mobilenet, self).__init__(input_nc, output_nc)
        self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
        self.backbone2 = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#

        self.aspp = ASPP(dim_in=320, dim_out=256, rate=16 // downsample_factor)
        self.aspp2 = ASPP(dim_in=320, dim_out=256, rate=16 // downsample_factor)
        self.se1 = Denosing(24, 24)
        self.se2 = Denosing(24, 24)
        self.la1 = LAAFNet(24, 24)
        self.la2 = LAAFNet(64, 64)
        self.la3 = LAAFNet(256, 256)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(24 + 24 + 64 + 64 + 256 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

        self.vit1 = StarNet(32, [3, 3, 12, 5])
        self.vit2 = StarNet(32, [3, 3, 12, 5])


    def forward(self, x1, x2):
        low_level_features, middle_features, x1 = self.backbone(x1)
        low_level_features2, middle_features2, x2 = self.backbone2(x2)

        low_level_features = self.se1(low_level_features)
        low_level_features2 = self.se2(low_level_features2)

        x1 = self.aspp(x1)
        x2 = self.aspp2(x2)

        fl = self.la1(low_level_features, low_level_features2)
        fm = self.la2(middle_features, middle_features2)
        fd = self.la3(x1, x2)

        fm = F.interpolate(fm, size=(fl.size(2), fl.size(3)), mode='bilinear', align_corners=True)
        fd = F.interpolate(fd, size=(fl.size(2), fl.size(3)), mode='bilinear', align_corners=True)

        x = self.cat_conv(torch.cat([fl, fm, fd], dim=1))

        x = self.cls_conv(x)

        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
        return x


class StarNets4(ResNet):
    def __init__(self, input_nc, output_nc, downsample_factor=8, in_channels=320, low_level_channels=24, num_classes=2,
                 pretrained=False):
        super(StarNets4, self).__init__(input_nc, output_nc)
        self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)


        self.aspp = ASPP(dim_in=256, dim_out=256, rate=16 // downsample_factor)
        self.aspp2 = ASPP(dim_in=256, dim_out=256, rate=16 // downsample_factor)
        self.se1 = Denosing(64, 64)
        self.se2 = Denosing(64, 64)

        self.la1 = LAAFNet(64, 64)
        self.la2 = LAAFNet(128, 128)
        self.la3 = LAAFNet(256, 256)
        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(64 + 64 + 128 + 128 + 256 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )


        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

        self.vit1 = StarNet(32, [3, 3, 12, 5])
        self.vit2 = StarNet(32, [3, 3, 12, 5])


    def forward(self, x1, x2):

        '''
        starnet
        '''

        x10,x11,x12,x13 = self.vit1(x1)

        x20,x21,x22,x23 = self.vit2(x2)

        x11 = self.se1(x11)
        x21 = self.se1(x21)

        x13 = self.aspp(x13)
        x23 = self.aspp2(x23)

        # fx0 = self.cbamlowlow(x10,x20)
        fx1 = self.la1(x11,x21)
        fx2 = self.la2(x12, x22)
        fx3 = self.la3(x13, x23)

        fx2 = F.interpolate(fx2, size=(fx1.size(2), fx1.size(3)), mode='bilinear', align_corners=True)
        fx3 = F.interpolate(fx3, size=(fx1.size(2), fx1.size(3)), mode='bilinear', align_corners=True)
        out1 = self.cat_conv(torch.cat([fx1, fx2, fx3], dim=1))

        out1 = F.interpolate(out1, size=(512, 512), mode='bilinear', align_corners=True)
        out1 = self.cls_conv(out1)
        return out1

from .vit import SETR
class VIT(ResNet):
    def __init__(self, input_nc, output_nc,):
        super(VIT,self).__init__(input_nc, output_nc,)

        self.aspp = ASPP(dim_in=64, dim_out=64, rate=16 // 8)
        self.aspp2 = ASPP(dim_in=64, dim_out=64, rate=16 // 8)
        self.se1 = Denosing(64, 16)
        self.se2 = Denosing(64, 16)

        self.la1 = LAAFNet(64, 64)
        self.la2 = LAAFNet(64, 64)
        self.la3 = LAAFNet(64, 64)
        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#

        self.cat_conv = nn.Sequential(
            nn.Conv2d(64 + 64 + 64 + 64 + 64 + 64, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

        self.cls_conv = nn.Conv2d(256, 2, 1, stride=1)

        self.vit1 = SETR(num_classes=64, image_size=256, patch_size=256 // 16, dim=1024, depth=24, heads=16, mlp_dim=2048,
                   out_indices=(9, 14, 19, 23)).cuda()
        self.vit2 = SETR(num_classes=64, image_size=256, patch_size=256 // 16, dim=1024, depth=24, heads=16, mlp_dim=2048,
                   out_indices=(9, 14, 19, 23)).cuda()


        self.cls_conv = nn.Conv2d(256, 2, 1, stride=1)

    def forward(self,x1,x2):

        x1 = self.vit1(x1)
        x2 = self.vit2(x2)

        x11,x12,x13 = x1[-3],x1[-2],x1[-1]
        x21, x22, x23 = x2[-3], x2[-2], x2[-1]

        x11 = self.se1(x11)
        x21 = self.se1(x21)

        x13 = self.aspp(x13)
        x23 = self.aspp2(x23)

        # fx0 = self.cbamlowlow(x10,x20)
        fx1 = self.la1(x11, x21)
        fx2 = self.la2(x12, x22)
        fx3 = self.la3(x13, x23)

        fx2 = F.interpolate(fx2, size=(fx1.size(2), fx1.size(3)), mode='bilinear', align_corners=True)
        fx3 = F.interpolate(fx3, size=(fx1.size(2), fx1.size(3)), mode='bilinear', align_corners=True)
        out1 = self.cat_conv(torch.cat([fx1, fx2, fx3], dim=1))

        out1 = F.interpolate(out1, size=(256, 256), mode='bilinear', align_corners=True)
        out1 = self.cls_conv(out1)
        return out1


