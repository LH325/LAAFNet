import torch
import torch.nn as nn

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, targets):
        batch_size = features.size(0)
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature
        targets = targets.view(-1, 1)
        positive_mask = torch.eq(targets, targets.t()).float()
        negative_mask = 1 - positive_mask

        positive_loss = torch.exp(similarity_matrix) * positive_mask
        negative_loss = torch.exp(similarity_matrix) * negative_mask

        positive_loss = -torch.log(positive_loss.sum(1) / positive_mask.sum(1))
        negative_loss = -torch.log(negative_loss.sum(1) / negative_mask.sum(1))

        loss = positive_loss + negative_loss
        loss = loss / batch_size

        return loss.mean()
a = torch.randint(0,2,(4,40))
b = torch.randn((4,40))
print(a.shape)
print(b.shape)
I = InfoNCELoss()
l = I(a,b)
print(l)