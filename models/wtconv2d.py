import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift


class HighFreqFeatureExtractor:
    def __init__(self, ratio=0.2):
        """
        初始化高频特征提取器
        :param ratio: 高频成分保留比例
        """
        self.ratio = ratio

    def extract_high_freq(self, img):
        """
        使用傅里叶变换提取图像的高频成分
        :param img: 输入图像 (C, H, W)
        :return: 高频成分图像 (C, H, W)
        """
        # 转换为复数张量
        img_complex = torch.view_as_complex(torch.stack([img, torch.zeros_like(img)], dim=-1))

        # 傅里叶变换
        fft_img = fft2(img_complex)
        fft_img_shifted = fftshift(fft_img, dim=(-2, -1))

        # 创建高频掩码
        C, H, W = img.shape
        mask = torch.ones((H, W), device=img.device)
        center_h, center_w = H // 2, W // 2
        radius_h, radius_w = int(H * self.ratio / 2), int(W * self.ratio / 2)

        mask[center_h - radius_h:center_h + radius_h, center_w - radius_w:center_w + radius_w] = 0

        # 应用掩码提取高频成分
        high_freq_shifted = fft_img_shifted * mask

        # 反变换回空间域
        high_freq = ifft2(ifftshift(high_freq_shifted, dim=(-2, -1)))
        high_freq = torch.abs(high_freq)  # 取模

        return high_freq


class HighFreqMSELoss(nn.Module):
    def __init__(self, ratio=0.2):
        """
        高频特征均方误差损失
        :param ratio: 高频成分保留比例
        """
        super(HighFreqMSELoss, self).__init__()
        self.extractor = HighFreqFeatureExtractor(ratio)
        self.mse_loss = nn.MSELoss()

    def forward(self, rgb_img, nir_img):
        """
        计算RGB和NIR图像高频特征的MSE损失
        :param rgb_img: RGB图像 (C, H, W) 或 (B, C, H, W)
        :param nir_img: NIR图像 (C, H, W) 或 (B, C, H, W)
        :return: 高频特征MSE损失
        """
        # 如果是批量数据，则逐个处理
        if rgb_img.dim() == 4:
            loss = 0
            for i in range(rgb_img.shape[0]):
                rgb_high = self.extractor.extract_high_freq(rgb_img[i])
                nir_high = self.extractor.extract_high_freq(nir_img[i])
                loss += self.mse_loss(rgb_high, nir_high)
            return loss / rgb_img.shape[0]
        else:
            rgb_high = self.extractor.extract_high_freq(rgb_img)
            nir_high = self.extractor.extract_high_freq(nir_img)
            return self.mse_loss(rgb_high, nir_high)


# 使用示例
if __name__ == "__main__":
    # 假设我们有一批RGB和NIR图像 (B, C, H, W)
    batch_size = 4
    channels = 3  # 对于NIR可能是1通道
    height, width = 256, 256

    # 创建模拟数据
    rgb_images = torch.rand(batch_size, channels, height, width)
    nir_images = torch.rand(batch_size, 1, height, width).expand(-1, channels, -1, -1)  # 扩展到3通道

    # 初始化损失函数
    loss_fn = HighFreqMSELoss(ratio=0.3)

    # 计算损失
    loss = loss_fn(rgb_images, nir_images)
    print(f"High Frequency MSE Loss: {loss.item()}")