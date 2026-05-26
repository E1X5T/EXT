import torch.nn as nn
import torch
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """注意力融合两个同一模态的特征（例如 shallow + deep）"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 16, 1)
        self.conv2 = nn.Conv2d(channels // 16, channels, 1)
        self.bn = nn.BatchNorm2d(channels // 16)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat1, feat2):
        fused = feat1 + feat2

        local = self.conv1(fused)
        local = F.relu(self.bn(local))
        local = self.conv2(local)

        global_feat = self.gap(fused)
        global_feat = self.conv1(global_feat)
        global_feat = F.relu(self.bn(global_feat))
        global_feat = self.conv2(global_feat)

        M = torch.sigmoid(local + global_feat)
        fm = M * feat1 + (1 - M) * feat2
        return fm
