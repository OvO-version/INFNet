import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.convpoint = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        y = self.conv(x)
        x = self.convpoint(x)
        x = y + x
        return x


class ViT(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embed_dim, num_heads, depth):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1)
        num_patches = (img_size // patch_size) ** 2
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        self.transformer_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True) for _ in range(depth)]
        )

        self.proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.positional_encoding

        features = []  # 用于存储每层的输出
        for layer in self.transformer_layers:
            x = layer(x)
            features.append(x)  # 存储 Transformer 层的输出

        num_patches = x.size(1)
        patches_per_dim = int(num_patches ** 0.5)
        x = x.transpose(1, 2).view(batch_size, self.embed_dim, patches_per_dim, patches_per_dim)
        x = self.proj(x)

        fused_features = []
        for f in features:
            f = f.transpose(1, 2).view(batch_size, self.embed_dim, patches_per_dim, patches_per_dim)
            fused_features.append(self.proj(f))  # 投影到 CNN 维度

        return x, fused_features  # 返回最终输出和所有 Transformer 层的特征



class LocalGlobalFusion(nn.Module):
    def __init__(self, cnn_channels, vit_channels):
        super(LocalGlobalFusion, self).__init__()
        self.cnn_to_vit = nn.Conv2d(cnn_channels, vit_channels, kernel_size=1)
        self.vit_to_cnn = nn.Conv2d(vit_channels * 2, cnn_channels, kernel_size=1)
        self.attn = nn.Sequential(
            nn.Conv2d(cnn_channels//2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_cnn, x_vit):
        cnn_guidance = self.cnn_to_vit(x_cnn)
        vit_guidance = self.vit_to_cnn(x_vit)

        attn_map = self.attn(cnn_guidance)
        enhanced_vit = x_vit * attn_map
        enhanced_cnn = x_cnn + vit_guidance

        return enhanced_cnn, enhanced_vit


class CorrelationBranch(nn.Module):
    def __init__(self, cnn_channels, vit_channels):
        super(CorrelationBranch, self).__init__()
        self.vit_projection = nn.Conv2d(512, cnn_channels, kernel_size=1)

    def forward(self, x_cnn, x_vit):
        x_vit_resized = self.vit_projection(x_vit)

        if x_cnn.size(2) != x_vit_resized.size(2) or x_cnn.size(3) != x_vit_resized.size(3):
            x_vit_resized = F.interpolate(x_vit_resized, size=(x_cnn.size(2), x_cnn.size(3)), mode='bilinear',
                                          align_corners=False)

        cosine_sim_map = self.cosine_similarity(x_cnn, x_vit_resized)
        combined_sim_map = cosine_sim_map
        return combined_sim_map

    def cosine_similarity(self, x_cnn, x_vit):
        x_cnn_flat = x_cnn.view(x_cnn.size(0), x_cnn.size(1), -1)  # (B, C, N)
        x_vit_flat = x_vit.view(x_vit.size(0), x_vit.size(1), -1)  # (B, C, N)

        # 计算每个空间位置的L2范数，dim=1（通道维度）
        norm_cnn = torch.norm(x_cnn_flat, p=2, dim=1, keepdim=True)  # (B, 1, N)
        norm_vit = torch.norm(x_vit_flat, p=2, dim=1, keepdim=True)  # (B, 1, N)

        # 点积计算
        cosine_sim = torch.bmm(x_cnn_flat.transpose(1, 2), x_vit_flat)  # (B, N, N)
        # 归一化
        cosine_sim = cosine_sim / (norm_cnn.transpose(1, 2) @ norm_vit + 1e-8)  # (B, N, N)

        # 取对角线元素作为对应位置的相似度
        diagonal_sim = cosine_sim.diagonal(dim1=1, dim2=2)  # (B, N)
        cosine_sim_map = diagonal_sim.view(x_cnn.size(0), 1, x_cnn.size(2), x_cnn.size(3))  # (B, 1, H, W)

        return cosine_sim_map


class ONet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512],
                 vit_img_size=224, vit_patch_size=16, vit_embed_dim=256, vit_heads=4, vit_depth=4):
        super(ONet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.vit_proj_1 = nn.Conv2d(vit_embed_dim * 2, features[2], kernel_size=1)
        self.vit_proj_2 = nn.Conv2d(vit_embed_dim * 2, features[1], kernel_size=1)
        self.vit_proj_3 = nn.Conv2d(vit_embed_dim * 2, features[0], kernel_size=1)
        self.vit_proj_4 = nn.Conv2d(vit_embed_dim * 2, features[0]//2, kernel_size=1)

        self.down1 = DoubleConv(in_channels, features[0])
        self.down2 = DoubleConv(features[0], features[1])
        self.down3 = DoubleConv(features[1], features[2])
        self.down4 = DoubleConv(features[2], features[3])

        self.vit = ViT(features[3], vit_img_size, vit_patch_size, vit_embed_dim, vit_heads, vit_depth)

        self.lg_fusion = LocalGlobalFusion(features[3], vit_embed_dim)
        self.corr_branch = CorrelationBranch(cnn_channels=features[3], vit_channels=vit_embed_dim)

        self.up1 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(features[3] * 2, features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(features[3], features[1])
        self.up3 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(features[2], features[0])
        self.up4 = nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(160, features[0])

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        down1 = self.down1(x)
        x = self.pool(down1)
        down2 = self.down2(x)
        x_2 = self.pool(down2)
        down3 = self.down3(x_2)
        x_3 = self.pool(down3)
        down4 = self.down4(x_3)
        x_cnn = self.pool(down4)

        x_vit, vit_features = self.vit(x_cnn)

        corr_attn = self.corr_branch(x_cnn, x_vit)
        x_cnn, x_vit = self.lg_fusion(x_cnn, x_vit)
        x_fused = x_cnn * (1 + corr_attn)

        # 上采样 ViT 特征，以匹配 CNN 层的尺寸
        vit_features_resized = [
            F.interpolate(f, size=(down4.size(2), down4.size(3)), mode='bilinear', align_corners=False) for f in
            vit_features
        ]
        x_vit_resized_1 = self.vit_proj_1(vit_features_resized[-1])
        x_vit_resized_2 = self.vit_proj_2(vit_features_resized[-2])
        x_vit_resized_3 = self.vit_proj_3(vit_features_resized[-3])
        x_vit_resized_4 = self.vit_proj_4(vit_features_resized[-4])


        # 使用上采样后的 ViT 特征进行跳连，调整尺寸匹配
        x = self.up1(x_fused)
        x = torch.cat([x, down4, x_vit_resized_1], dim=1)  # 跳连最后一层 ViT 特征
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, down3,
                       F.interpolate(x_vit_resized_2, size=(down3.size(2), down3.size(3)), mode='bilinear',
                                     align_corners=False)], dim=1)  # 调整 viT 特征尺寸
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, down2,
                       F.interpolate(x_vit_resized_3, size=(down2.size(2), down2.size(3)), mode='bilinear',
                                     align_corners=False)], dim=1)  # 调整 viT 特征尺寸
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, down1,
                       F.interpolate(x_vit_resized_4, size=(down1.size(2), down1.size(3)), mode='bilinear',
                                     align_corners=False)], dim=1)  # 调整 viT 特征尺寸
        x = self.conv_up4(x)

        return self.final_conv(x)


def test():
    x = torch.randn(4, 3, 224, 224)
    model = ONet(in_channels=3, out_channels=1)
    preds = model(x)
    flops, params = profile(model, inputs=(x,))
    print(f'Flops: {flops}, params: {params}')
    print(preds.shape)


if __name__ == "__main__":
    test()
