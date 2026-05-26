import torch
import torch.nn as nn
# =========================================================
# 1. 基础残差块 (无任何注意力)
# =========================================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

# =========================================================
# 2. 单模态主干网分支 (纯净版)
# =========================================================
class FeatureExtractionBranch(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureExtractionBranch, self).__init__()

        # 现代瘦身版 Stem (使用 3x3 卷积保护细粒度纹理)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        # 注意：此处没有任何 MDFA 或其他增强模块

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        curr_channels = in_channels
        for s in strides:
            layers.append(BasicBlock(curr_channels, out_channels, stride=s))
            curr_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # 返回干净的 [B, 256, 14, 14]

# =========================================================
# 3. 双模态基线分类网络 (暴力 Cat 融合)
# =========================================================
class DualModalBaselineNet(nn.Module):
    def __init__(self, num_classes):
        super(DualModalBaselineNet, self).__init__()
        self.vein_branch = FeatureExtractionBranch(in_channels=3)
        self.fingerprint_branch = FeatureExtractionBranch(in_channels=3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # [256 + 256 = 512] -> 全连接层输入为 512
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_vein, x_fingerprint):
        feat_vein = self.vein_branch(x_vein)
        feat_fingerprint = self.fingerprint_branch(x_fingerprint)

        # 暴力拼接
        fused_feat = torch.cat((feat_vein, feat_fingerprint), dim=1)

        pooled_feat = torch.flatten(self.global_pool(fused_feat), 1)
        return self.classifier(pooled_feat)






























# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # =========================================================
# # 1. 基础残差块 (与基线网络保持完全一致)
# # =========================================================
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         identity = self.shortcut(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += identity
#         return self.relu(out)
#
#
# # =========================================================
# # 2. 完全修复版：多尺度空洞融合注意力模块 (MDFA)
# # =========================================================
# class tongdao(nn.Module):
#     """修复版通道注意力：严格遵循 C -> C/2 -> C 的瓶颈校准结构"""
#
#     def __init__(self, in_channel):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # 降维到 C/2，再升维回 C
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channel // 2, in_channel, kernel_size=1, bias=False),
#             nn.Sigmoid()  # 对应图纸标注的 Sigmoid 激活
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.fc(y)
#         return x * y  # 逐通道特征重校准
#
#
# class kongjian(nn.Module):
#     """空间注意力机制：保持不变"""
#
#     def __init__(self, in_channel):
#         super().__init__()
#         self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
#         self.norm = nn.Sigmoid()
#
#     def forward(self, x):
#         y = self.Conv1x1(x)
#         y = self.norm(y)
#         return x * y
#
#
# class hebing(nn.Module):
#     """特征合并：严格遵循图纸右侧的蓝色 'Add' 圆圈，使用元素相加"""
#
#     def __init__(self, in_channel):
#         super().__init__()
#         self.tongdao = tongdao(in_channel)
#         self.kongjian = kongjian(in_channel)
#
#     def forward(self, U):
#         U_kongjian = self.kongjian(U)
#         U_tongdao = self.tongdao(U)
#         return U_tongdao + U_kongjian  # 元素相加融合双重注意力
#
#
# class MDFA(nn.Module):
#     def __init__(self, dim_in, dim_out, bn_mom=0.1):
#         super(MDFA, self).__init__()
#         # ⚠️ 关键适配：由于嵌入在 Layer3 后（特征图为 14x14），
#         # 将空洞率调整为更适合当前尺寸的 2, 4, 6，防止外围采样全是零 Padding。
#         rate1, rate2, rate3 = 2, 4, 6
#
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         # 【分支 2：横向条形卷积】专门贴合手指静脉的横向血管，rate=2 完美适应 14x14 尺寸
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out,
#                       kernel_size=(1, 5),
#                       stride=1,
#                       padding=(0, 2),
#                       dilation=(1, 1),  # 也可以尝试开启轻量空洞 (1, 2)
#                       bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         # 【分支 3：纵向条形卷积】专门贴合手指静脉的纵向脊线/血管
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out,
#                       kernel_size=(5, 1),
#                       stride=1,
#                       padding=(2, 0),
#                       dilation=(1, 1),  # 也可以尝试开启轻量空洞 (2, 1)
#                       bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         # 【分支 4：缩小版正方形空洞】捕捉全局拓扑，防止越界
#         self.branch4 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=2, dilation=2, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         # 第五分支：全局上下文提取
#         self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
#         self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
#         self.branch5_relu = nn.ReLU(inplace=True)
#
#         # 注意力校准组件
#         self.Hebing = hebing(in_channel=dim_out * 5)
#
#         # 最终整合与降维
#         self.conv_cat = nn.Sequential(
#             nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         [b, c, row, col] = x.size()
#
#         # 运行前四个多尺度多感受野分支
#         conv1x1 = self.branch1(x)
#         conv3x3_1 = self.branch2(x)
#         conv3x3_2 = self.branch3(x)
#         conv3x3_3 = self.branch4(x)
#
#         # 第五分支：全局上下文特征提取
#         global_feature = torch.mean(x, 2, True)
#         global_feature = torch.mean(global_feature, 3, True)
#         global_feature = self.branch5_conv(global_feature)
#         global_feature = self.branch5_bn(global_feature)
#         global_feature = self.branch5_relu(global_feature)
#
#         # ⚠️ 核心修复：改用 'nearest' 模式进行安全上采样，彻底避免 1x1 特征图引发的插值报错
#         global_feature = F.interpolate(global_feature, (row, col), mode='nearest')
#
#         # 拼接多尺度特征
#         feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#
#         # 施加通道与空间双重校准
#         larry = self.Hebing(feature_cat)
#         larry_feature_cat = larry * feature_cat
#
#         # 降维输出
#         result = self.conv_cat(larry_feature_cat)
#         return result
#
#
# # =========================================================
# # 3. 单模态主干网分支 (无缝嵌入 MDFA 模块)
# # =========================================================
# class FeatureExtractionBranch(nn.Module):
#     def __init__(self, in_channels=3):
#         super(FeatureExtractionBranch, self).__init__()
#
#         # 精简版浅层 Stem 层 (3x3 卷积保护初始细粒度纹理)
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#
#         self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
#         self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
#         self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
#
#         # 🌟 精准嵌入：插在 Layer 3 之后，提取高层多尺度语义
#         # 输入 256 通道，输出 256 通道，不改变张量空间尺寸
#         self.mdfa = MDFA(dim_in=256, dim_out=256)
#
#     def _make_layer(self, in_channels, out_channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         curr_channels = in_channels
#         for s in strides:
#             layers.append(BasicBlock(curr_channels, out_channels, stride=s))
#             curr_channels = out_channels
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#
#         # 特征在进入融合前经过 MDFA 注意力增强
#         x = self.mdfa(x)
#         return x
#
#
# # =========================================================
# # 4. 双模态整合分类网络
# # =========================================================
# class DualModalBaselineNet(nn.Module):
#     def __init__(self, num_classes):
#         super(DualModalBaselineNet, self).__init__()
#         self.vein_branch = FeatureExtractionBranch(in_channels=3)
#         self.fingerprint_branch = FeatureExtractionBranch(in_channels=3)
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#
#         # 两个分支各输出 256 通道，直接拼接后依旧为 512，完美兼容原先的分类头
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(512, num_classes)
#         )
#
#     def forward(self, x_vein, x_fingerprint):
#         feat_vein = self.vein_branch(x_vein)
#         feat_fingerprint = self.fingerprint_branch(x_fingerprint)
#
#         fused_feat = torch.cat((feat_vein, feat_fingerprint), dim=1)
#         pooled_feat = torch.flatten(self.global_pool(fused_feat), 1)
#         return self.classifier(pooled_feat)