import torch
import random
import torch.nn.functional as F
import numpy as np


def random_select_pixel(feature_num, select_num):
    feature = feature_num.t()
    pixel_num = feature.shape[0]
    min_select = min(pixel_num, select_num)
    random_index = random.sample(range(0, pixel_num), min_select)
    selected_feature = feature[random_index, :]
    normalized_feature = torch.nn.functional.normalize(selected_feature, p=2, dim=1)
    return normalized_feature


def random_select_half_hard_pixel(feature_clone_2, easy_index, hard_index, select_num, hard_pixel_prop):
    # 将特征展平处理
    C, B, H, W = feature_clone_2.shape


    feature_flatten = feature_clone_2.reshape(C, -1)  # (C, B*H*W)

    # 获取展平后的索引
    easy_indices = easy_index.reshape(-1).bool()
    hard_indices = hard_index.reshape(-1).bool()


    # 提取对应特征
    easy_feature = feature_flatten[:, easy_indices].t()  # (num_easy, C)
    hard_feature = feature_flatten[:, hard_indices].t()  # (num_hard, C)

    hard_num = hard_feature.size(0)
    easy_num = easy_feature.size(0)

    if easy_num==0 or hard_num==0:
        return None
    else:
        hard_pixel_prop = float(hard_pixel_prop)
        hard_prop_num = int(select_num * hard_pixel_prop)
        easy_prop_num = int(select_num * (1 - hard_pixel_prop))

        # 调整采样数量逻辑
        if hard_num + easy_num < select_num:
            selected_feature = torch.cat([easy_feature, hard_feature], dim=0)
        else:
            if hard_num < hard_prop_num or easy_num < easy_prop_num:
                hard_select_num = min(hard_num, select_num)
                easy_select_num = select_num - hard_select_num
                if easy_select_num > easy_num:
                    easy_select_num = easy_num
                    hard_select_num = select_num - easy_select_num
            else:
                hard_select_num = hard_prop_num
                easy_select_num = easy_prop_num

            # 随机采样
            if easy_num > 0:
                random_easy_index = random.sample(range(easy_num), easy_select_num)
                selected_easy = easy_feature[random_easy_index]
            else:
                selected_easy = torch.Tensor()

            if hard_num > 0:
                random_hard_index = random.sample(range(hard_num), hard_select_num)
                selected_hard = hard_feature[random_hard_index]
            else:
                selected_hard = torch.Tensor()


            selected_feature = torch.cat([selected_easy, selected_hard], dim=0)

        # 特征归一化
        normalized_feature = F.normalize(selected_feature, p=2, dim=1)
    return normalized_feature


def js_div(p_logits, q_logits, get_softmax=True):
    KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_logits, dim=1)
        q_output = F.softmax(q_logits, dim=1)
    else:
        p_output = p_logits
        q_output = q_logits
    p_output = p_output.to('cuda:0')
    q_output = q_output.to('cuda:0')
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


def cal_pixel_contrast_all_relation_loss(pixel_feature, temperature):
    cat_feature = torch.cat(pixel_feature, dim=0)
    mask_matrix = torch.zeros((len(cat_feature), len(cat_feature)))
    start_idx = 0
    for feat in pixel_feature:
        end_idx = start_idx + feat.size(0)
        mask_matrix[start_idx:end_idx, start_idx:end_idx] = 1
        start_idx = end_idx

    anchor_dot_contrast = torch.matmul(cat_feature, cat_feature.T) / temperature

    exp_anchor = F.softmax(anchor_dot_contrast, dim=1)
    mask_normalized = mask_matrix / mask_matrix.sum(dim=1, keepdim=True)
    loss = js_div(exp_anchor, mask_normalized, get_softmax=False)
    return loss

def pixel2pixel(feature, label, pre_mask):
    # 示例数据
    label = label.squeeze(dim=1)
    source_label_downsampled = label  # 假设不需要下采样

    # 参数设置
    class Args:
        random_select_num = 500
        hard_pixel_prop = 0.5

    args = Args()

    feature_clone = feature.clone()
    feature_clone_2 = feature_clone.transpose(1, 0)  # (24, 256, 512, 512)

    # 计算预测类别
    _, pre_mask_max = torch.max(pre_mask, dim=1)
    wrong_index = pre_mask_max != source_label_downsampled
    correct_index = pre_mask_max == source_label_downsampled

    pixel_contrast_feature = []
    for label_id in range(2):
        # 获取当前类别的像素索引
        index = label == label_id

        # 处理特征选择
        C, B, H, W = feature_clone_2.shape
        # feature_flatten = feature_clone_2.reshape(C, -1)
        # index_flatten = index.reshape(-1)
        # feature_num = feature_flatten[:, index_flatten].t()

        # 获取难易样本索引
        hard_index = index & wrong_index
        easy_index = index & correct_index

        # 选择特征
        current_feature = random_select_half_hard_pixel(
            feature_clone_2, easy_index, hard_index,
            args.random_select_num, args.hard_pixel_prop
        )
        if current_feature == None:
            contrast_loss = 0
            return contrast_loss
        else:
            pixel_contrast_feature.append(current_feature)

    # 计算对比损失
    contrast_loss = cal_pixel_contrast_all_relation_loss(pixel_contrast_feature, temperature=0.1)
    # print(f"Pixel Contrastive Loss: {contrast_loss.item()}")
    return contrast_loss
if __name__ == '__main__':
    feature = torch.rand((8, 256, 512, 512))
    label = torch.randint(0, 2, (8, 512, 512))

    # 模拟预测结果
    pre_mask = torch.randn(8, 2, 512, 512)
    pixel2pixel(feature,label,pre_mask)