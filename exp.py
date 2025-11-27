import torch
import torch.nn.functional as F

# 假设A和B是两个形状相同的张量，每个元素点都是一个d维的向量
# 例如，A和B的形状都是(batch_size, num_points, d)
A = torch.randn(10, 5, 3)  # 10个样本，每个样本有5个点，每个点是3维的
B = torch.randn(10, 5, 3)  # 同上

# 计算分子部分：A和B的点积
dot_product = torch.bmm(A, B.transpose(1, 2))

# 计算分母部分：A和B的范数的乘积
norm_A = torch.norm(A, dim=2, keepdim=True)
norm_B = torch.norm(B, dim=2, keepdim=True)

# 计算余弦相似度
cosine_similarity = dot_product / (norm_A * norm_B.transpose(1, 2))
print(cosine_similarity.shape)
# cosine_similarity的形状是(batch_size, num_points, num_points)
# 表示每个样本中num_points个点与其他点的余弦相似度