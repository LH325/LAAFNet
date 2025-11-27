import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np



from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def focal_loss(input,
               target,
               weight=None,
               gamma=2.0,
               alpha=None,
               ignore_index=-100,
               reduction='mean'):
    """
    Focal Loss for multi-class (支持分类 / 语义分割).
    参数基本对齐 torch.nn.functional.cross_entropy。

    input:  logits, 形状 [N, C] 或 [N, C, H, W]
    target: 标签,  形状 [N] 或 [N, H, W]，里面是类别索引
    weight: 每个类别的权重, 和 cross_entropy 一样
    gamma:  Focal Loss 的 gamma, 通常取 2.0
    alpha:  类别平衡因子，可以是:
            - None: 不使用 alpha
            - float 标量: 整体缩放正样本
            - Tensor(shape=[C]): 每个类别一个 α
    ignore_index: 忽略的标签索引
    reduction: 'none' | 'mean' | 'sum'
    """

    # 先算普通 CE，但不做 reduction，得到每个像素/样本的 loss
    ce_loss = F.cross_entropy(
        input=input,
        target=target,
        weight=weight,
        ignore_index=ignore_index,
        reduction='none'
    )  # shape: [N] or [N,H,W]

    # p_t = exp(-CE) = 对应真实标签的概率
    # 因为 CE = -log(p_t)
    pt = torch.exp(-ce_loss)  # same shape as ce_loss

    # Focal Loss 的调制因子: (1 - p_t)^gamma
    focal_factor = (1 - pt) ** gamma

    # 处理 alpha（类别平衡）
    if alpha is not None:
        # alpha 可以是标量或 Tensor([C])
        if isinstance(alpha, (float, int)):
            alpha_factor = torch.ones_like(ce_loss) * float(alpha)
        else:
            # alpha 是 per-class tensor，按 target 索引
            # 需要先处理 ignore_index
            if alpha.device != input.device:
                alpha = alpha.to(input.device)

            alpha_factor = torch.ones_like(ce_loss)
            valid_mask = (target != ignore_index)
            if valid_mask.any():
                alpha_factor[valid_mask] = alpha[target[valid_mask]]
        focal = alpha_factor * focal_factor * ce_loss
    else:
        focal = focal_factor * ce_loss

    # 处理 reduction
    if reduction == 'mean':
        # 忽略 ignore_index 对应的位置
        if ignore_index is not None:
            valid_mask = (target != ignore_index)
            if valid_mask.any():
                return focal[valid_mask].mean()
            else:
                return focal.mean()  # 退化情况
        else:
            return focal.mean()
    elif reduction == 'sum':
        if ignore_index is not None:
            valid_mask = (target != ignore_index)
            return focal[valid_mask].sum()
        else:
            return focal.sum()
    else:  # 'none'
        return focal
class PiCoPixelContrastLoss(nn.Module):
    """
    PiCo 像素–像素对比损失 (简化版，无硬样本策略 & 合成，只实现核心 InfoNCE)
    - feats:  [B, D, H, W]
    - labels: [B, H, W]
    - 可选 memory_feats:  [N_mem, D]
            memory_labels: [N_mem]

    正样本: 相同类别的像素
    负样本: 其他类别的像素
    """

    def __init__(
        self,
        temperature=0.1,
        base_temperature=0.07,
        ignore_label=255,
        max_views_per_class=100,   # 每类最多采多少个像素参与对比
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_label = ignore_label
        self.max_views_per_class = max_views_per_class

    @torch.no_grad()
    def _sample_pixels(self, feats, labels):
        """
        从当前 batch 中按类别采像素：
        feats:  [B, D, H, W]
        labels: [B, H, W]
        返回:
          - sampled_feats:  [N_sel, D]
          - sampled_labels: [N_sel]
        """
        device = feats.device
        B, D, H, W = feats.shape

        # [B, D, H, W] -> [B, H, W, D] -> [B*H*W, D]
        feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, D)  # [N_pix, D]
        labels_flat = labels.view(-1)                          # [N_pix]

        # 去掉 ignore_label
        valid_mask = labels_flat != self.ignore_label
        feats_flat = feats_flat[valid_mask]
        labels_flat = labels_flat[valid_mask]

        if feats_flat.numel() == 0:
            return None, None

        unique_classes = labels_flat.unique()
        sampled_feats = []
        sampled_labels = []

        for cls in unique_classes:
            c = int(cls.item())
            if c == self.ignore_label:
                continue
            cls_mask = labels_flat == c
            idxs = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
            if idxs.numel() == 0:
                continue

            # 随机子采样，最多 max_views_per_class 个像素
            if idxs.numel() > self.max_views_per_class:
                perm = torch.randperm(idxs.numel(), device=device)[: self.max_views_per_class]
                idxs = idxs[perm]

            sampled_feats.append(feats_flat[idxs])
            sampled_labels.append(labels_flat[idxs])

        if len(sampled_feats) == 0:
            return None, None

        sampled_feats = torch.cat(sampled_feats, dim=0)    # [N_sel, D]
        sampled_labels = torch.cat(sampled_labels, dim=0)  # [N_sel]
        return sampled_feats, sampled_labels

    def forward(
        self,
        pixel_feats,
        labels,
        memory_feats=None,
        memory_labels=None,
    ):
        """
        pixel_feats: [B, D, H, W]  (已经 L2 归一化最好)
        labels:      [B, H, W]

        memory_feats:  [N_mem, D] (可选)
        memory_labels: [N_mem]    (可选)
        """
        device = pixel_feats.device

        # 1) 从当前 batch 中采像素作为 anchor
        anchor_feats, anchor_labels = self._sample_pixels(pixel_feats, labels)
        if anchor_feats is None:
            # 没有有效像素，返回 0，避免 NaN
            return pixel_feats.sum() * 0.0

        # 2) 对比集合：可以只用当前 batch，也可以 + memory
        if (memory_feats is not None) and (memory_labels is not None):
            contrast_feats = torch.cat([anchor_feats, memory_feats.to(device)], dim=0)
            contrast_labels = torch.cat([anchor_labels, memory_labels.to(device)], dim=0)
        else:
            contrast_feats = anchor_feats
            contrast_labels = anchor_labels

        # 3) L2 归一化（如果外面没做，这里做一次）
        anchor_feats = F.normalize(anchor_feats, dim=1)
        contrast_feats = F.normalize(contrast_feats, dim=1)

        # 4) 相似度矩阵 logits: [N_anchor, N_contrast]
        logits = torch.div(
            torch.matmul(anchor_feats, contrast_feats.t()),
            self.temperature,
        )

        N_anchor = anchor_feats.shape[0]
        N_contrast = contrast_feats.shape[0]

        # 5) 构建正样本 mask: 同类为 1，异类为 0
        anchor_labels = anchor_labels.view(-1, 1)     # [N_anchor, 1]
        contrast_labels = contrast_labels.view(1, -1) # [1, N_contrast]
        mask = torch.eq(anchor_labels, contrast_labels).float().to(device)  # [N_anchor, N_contrast]

        # 6) 去掉“自己和自己”的对比
        # 如果对比集合里包含了 anchor 本身的特征（前 N_anchor 个），则屏蔽对角线部分
        logits_mask = torch.ones_like(mask, device=device)
        if N_contrast >= N_anchor:
            logits_mask[:, :N_anchor].fill_diagonal_(0)
        mask = mask * logits_mask  # 屏蔽 self-positive

        # 7) InfoNCE: log_softmax + 只在正样本上取期望
        exp_logits = torch.exp(logits) * logits_mask   # [N_anchor, N_contrast]
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 每个 anchor 的正样本数量
        pos_count = mask.sum(dim=1)  # [N_anchor]
        valid_anchor = pos_count > 0

        if valid_anchor.sum() == 0:
            # 没有任何 anchor 有正样本，返回 0 防止 NaN
            return pixel_feats.sum() * 0.0

        # 对每个 anchor，平均它的所有正样本 log_prob
        mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid_anchor] / pos_count[valid_anchor]

        # NCE loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """
    Computes cosine similarity between two tensors.
    """
    x1 = x1.float().cuda()
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, p=2, dim=1)
    w2 = torch.norm(x2, p=2, dim=1)
    return w12 / (w1 * w2 + eps)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Computes the triplet loss.

        Parameters:
        - anchor: Tensor of shape (batch_size, feature_dim)
        - positive: Tensor of shape (batch_size, feature_dim)
        - negative: Tensor of shape (batch_size, feature_dim)

        Returns:
        - loss: Scalar tensor representing the triplet loss.
        """
        # Compute cosine similarities

        positive = positive
        negative = negative
        pos_sim = cosine_similarity(anchor, positive)
        neg_sim = cosine_similarity(anchor, negative)

        # Compute triplet loss
        loss = F.relu(self.margin - pos_sim + neg_sim)

        # Return the mean loss over the batch
        return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵
        self.mse = nn.MSELoss()

    def contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        0.1)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (0.1 / 0.07) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        # z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        # z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        '''
        新版
        '''

        # representations = torch.stack([emb_i, emb_j], dim=1)  # repre: (2*bs, dim)
        # # loss = self.contrastive(representations,label_tif_tif)
        # T = representations.transpose(0, 1)
        # # T = representations.permute((0,2,1))
        # out = torch.mm(T, representations)
        #
        # # out = einsum(representations,T,'i j k, i k j -> i j')
        # out = torch.softmax(out, dim=-1)
        #
        # label = torch.FloatTensor([[1, 0], [0, 1]]).to('cuda:0')
        # loss = self.mse(out, label)
        '''
        旧版
        '''
        representations = torch.cat([emb_i, emb_j],dim=0)  # repre: (2*bs, dim)
        # loss = self.contrastive(representations,label_tif_tif)
        T = representations.transpose(0,1)
        # T = representations.permute((0,2,1))
        out = torch.mm(representations,T)

        # out = einsum(representations,T,'i j k, i k j -> i j')
        out = torch.softmax(out,dim=-1)

        label = torch.FloatTensor([[1,0],[0,1]]).to('cuda:0')
        loss = self.mse(out,label)
        '''
        其他版本
        '''
        # representations = torch.cat([emb_i, emb_j], dim=0)  # repre: (2*bs, dim)
        # T = representations.transpose(0,1)
        # out = torch.mm(representations,T)
        #
        # # out = torch.softmax(out,dim=-1)
        # label_tif_tif = torch.LongTensor([0,1]).to('cuda:0')
        # loss = F.cross_entropy(out,label_tif_tif)

        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
        #                                         dim=2)  # simi_mat: (2*bs, 2*bs)

        # sim_ij = torch.diag(similarity_matrix)[0]  # bs
        # sim_ji = torch.diag(similarity_matrix)[1]  # bs
        # positives = torch.stack([sim_ij, sim_ji], dim=0)  # 2*bs

        # nominator = torch.exp(positives / self.temperature)  # 2*bs
        # denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs
        #
        # loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        # loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

def Dice_loss(inputs, target, beta=1, smooth=1e-5):

    target = target.permute((0,2,3,1))
    target = torch.cat([target,target,target],dim=-1)
    if target.dim() == 3:
        target = torch.unsqueeze(target, dim=3)
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss
def Focal_Loss(inputs, target, cls_weights=None, num_classes=2, alpha=0.5, gamma=2):
    dice_loss = Dice_loss(inputs,target)
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    n, c, h, w = inputs.size()

    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    loss = loss+dice_loss
    return loss
def cross(input, target, weight=None, reduction='mean',ignore_index=255):

    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    # C = input.shape[1]

    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)

    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                    ignore_index=ignore_index, reduction=reduction)

def cross_entropy(input, target, K, weight=None, reduction='mean',ignore_index=255):

    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    # C = input.shape[1]
    # mse = nn.MSELoss()
    # k1,k2,k3,k4,k5,k6 = K
    # O1 = mse(k1,k2)
    # O2 = mse(k3, k4)
    # O3 = mse(k5, k6)
    # O = O1+O2+O3

    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)

    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    # contractive = ContrastiveLoss(batch_size=1)
    # y_t,_ = torch.max(input,dim=1)
    #
    # yindx = torch.argmax(input, dim=1)
    # yindx = yindx.view(input.shape[0],-1)
    #
    # y = y_t.view((input.shape[0],-1))
    #
    # t = target.view((target.shape[0],-1))
    # lossesc = 0
    #
    # for idx in range(input.shape[0]):
    #     # positive = torch.zeros((2, 100, 256),dtype=torch.float).cuda()
    #     classes = torch.unique(t[idx])
    #     classes = classes.detach().cpu().numpy()
    #     classes = list(classes)
    #
    #     if len(classes) == 1:
    #         continue
    #
    #     hard_w_indices = ((yindx[idx] == 1) & (t[idx] != 1)).nonzero()
    #     # easy_w_indices = ((yindx[idx] == 1) & (t[idx] == 1)).nonzero()
    #
    #     if hard_w_indices.shape[0] >50:
    #         perm = torch.randperm(50)
    #         hard_w_indices = hard_w_indices[perm[:50]]
    #         hard_water_sample = y[idx,hard_w_indices]
    #
    #     else:
    #         continue
    #     hard_b_indices = ((yindx[idx] == 0) & (t[idx] != 0)).nonzero()
    #
    #     if hard_b_indices.shape[0] > 50:
    #         perm = torch.randperm(50)
    #         hard_b_indices = hard_b_indices[perm[:50]]
    #         hard_background_sample = y[idx, hard_b_indices]
    #
    #     else:
    #         continue
    #
    #     hard_background_sample = hard_background_sample.view((1,-1))
    #     hard_water_sample = hard_water_sample.view((1,-1))
    #     lossc = contractive(hard_background_sample,hard_water_sample)
    #     lossesc += lossc
    #
    # if lossesc == 0:
    #     return F.cross_entropy(input=input, target=target, weight=weight,
    #                 ignore_index=ignore_index, reduction=reduction)
    # else:
    #
    #     loss = F.cross_entropy(input=input, target=target, weight=weight,
    #                     ignore_index=ignore_index, reduction=reduction) + 0.001 * lossesc
    #     return loss
    # pico = PiCoPixelContrastLoss()
    # picoloss = pico(input,target)

    # return F.cross_entropy(input=input, target=target, weight=weight,
    #                 ignore_index=ignore_index, reduction=reduction)
    return focal_loss(input=input, target=target, weight=weight,
                     ignore_index=ignore_index, reduction=reduction)

def DSPLoss(input, target, weight=None, reduction='mean',ignore_index=255):

    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    # C = input.shape[1]

    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)

    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    B,C,W,H = input.shape

    contractive = ContrastiveLoss(batch_size=1)

    y_t,_ = torch.max(input,dim=1)
    yindx = torch.argmax(input, dim=1)
    yindx = yindx.view(input.shape[0],-1)

    y = y_t.view((input.shape[0],-1))

    pred = input.view((B,C,W*H))
    pred = pred.permute((0,2,1))

    t = target.view((target.shape[0],-1))
    lossesc = 0

    for idx in range(input.shape[0]):
        # positive = torch.zeros((2, 100, 256),dtype=torch.float).cuda()
        classes = torch.unique(t[idx])
        classes = classes.detach().cpu().numpy()
        classes = list(classes)

        if len(classes) == 1:
            continue

        hard_w_indices = ((yindx[idx] == 1) & (t[idx] != 1)).nonzero()
        # easy_w_indices = ((yindx[idx] == 1) & (t[idx] == 1)).nonzero()

        if hard_w_indices.shape[0] >50:
            perm = torch.randperm(50)
            hard_w_indices = hard_w_indices[perm[:50]]
            hard_water_sample = pred[idx,hard_w_indices,:]


        else:
            continue
        hard_b_indices = ((yindx[idx] == 0) & (t[idx] != 0)).nonzero()

        if hard_b_indices.shape[0] > 50:
            perm = torch.randperm(50)
            hard_b_indices = hard_b_indices[perm[:50]]
            hard_background_sample = pred[idx, hard_b_indices,:]

        else:
            continue

        hard_water_sample = hard_water_sample.squeeze(1)
        hard_background_sample = hard_background_sample.squeeze(1)

        mean_X = torch.mean(hard_water_sample,dim=0)
        mean_Y = torch.mean(hard_background_sample,dim=0)
        hard_water_sample = hard_water_sample - mean_X
        hard_background_sample = hard_background_sample - mean_Y

        cov_water = torch.mm(hard_water_sample.T, hard_water_sample)/(hard_water_sample.size(0)-1)

        cov_background = torch.mm(hard_background_sample.T, hard_background_sample)/(hard_background_sample.size(0)-1)

        water_U, water_S , water_H = torch.linalg.svd(cov_water)
        background_U, background_S, background_H = torch.linalg.svd(cov_background)

        singular_values = water_S.diag()
        water_max_indices = torch.argmax(singular_values)
        # water_U = water_U[:,water_max_indices]
        # water_U = water_U.unsqueeze(1)
        # hard_water_sample = torch.matmul(hard_water_sample, water_U)
        lambdaw = singular_values[0,0] - singular_values[1,1]

        hard_water_sample = hard_water_sample[:,water_max_indices] * lambdaw

        singular_b_values = background_S.diag()
        background_max_indices = torch.argmax(singular_b_values)
        lambdaw = singular_values[0, 0] - singular_values[1, 1]
        hard_background_sample = hard_background_sample[:,background_max_indices] * lambdaw
        # background_U = background_U[:, background_max_indices]
        # background_U = background_U.unsqueeze(1)
        # hard_background_sample = torch.matmul(hard_background_sample, background_U)

        lossc = contractive(hard_background_sample, hard_water_sample)
        lossesc += lossc

    if lossesc == 0:
        return F.cross_entropy(input=input, target=target, weight=weight,
                    ignore_index=ignore_index, reduction=reduction)
    else:

        loss = F.cross_entropy(input=input, target=target, weight=weight,
                        ignore_index=ignore_index, reduction=reduction) + 0.001 * lossesc
        return loss
def DSPLoss_plus(input, target, weight=None, reduction='mean',ignore_index=255):

    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    # C = input.shape[1]

    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)

    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    contractive = ContrastiveLoss(batch_size=1)
    y_t, _ = torch.max(input, dim=1)
    y_t_2, _ = torch.min(input, dim=1)

    predmax = torch.argmax(input, dim=1)
    predmax = predmax.view(input.shape[0], -1)

    y = y_t.view((input.shape[0], -1))
    y2 = y_t_2.view((input.shape[0], -1))

    t = target.view((target.shape[0], -1))
    lossesc = 0

    for idx in range(input.shape[0]):
        # positive = torch.zeros((2, 100, 256),dtype=torch.float).cuda()
        classes = torch.unique(t[idx])
        classes = classes.detach().cpu().numpy()
        classes = list(classes)

        if len(classes) == 1:
            continue

        hard_w_indices = ((predmax[idx] == 1) & (t[idx] != 1)).nonzero()
        # easy_w_indices = ((yindx[idx] == 1) & (t[idx] == 1)).nonzero()

        if hard_w_indices.shape[0] > 50:
            perm = torch.randperm(50)
            hard_w_indices = hard_w_indices[perm[:50]]
            easy_w_indices = hard_w_indices
            hard_water_sample = y[idx, hard_w_indices]

            easy_water_sample = y2[idx,easy_w_indices]

        else:
            continue
        hard_b_indices = ((predmax[idx] == 0) & (t[idx] != 0)).nonzero()

        if hard_b_indices.shape[0] > 50:
            perm = torch.randperm(50)
            hard_b_indices = hard_b_indices[perm[:50]]
            easy_b_indices = hard_b_indices[perm[:50]]
            hard_background_sample = y[idx, hard_b_indices]

            easy_background_sample = y2[idx, easy_b_indices]

        else:
            continue

        hard_background_sample = hard_background_sample.view((1, -1))
        hard_water_sample = hard_water_sample.view((1, -1))

        easy_background_sample = easy_background_sample.view((1, -1))
        easy_water_sample = easy_water_sample.view((1, -1))

        lossc1 = contractive(hard_background_sample, hard_water_sample)
        lossc2 = contractive(easy_background_sample, easy_water_sample)

        lossc = lossc1 + lossc2
        lossesc += lossc

    if lossesc == 0:
        return F.cross_entropy(input=input, target=target, weight=weight,
                               ignore_index=ignore_index, reduction=reduction)
    else:

        loss = F.cross_entropy(input=input, target=target, weight=weight,
                               ignore_index=ignore_index, reduction=reduction) + 0.0005 * lossesc
        return loss