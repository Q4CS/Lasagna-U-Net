"""
[1] J. Qiu, Y. Chen, J. Zhao, Y. Huang, F. Wang, and Y. Li, ‘Lasagna-U-Net: Revisiting U-Shape Architecture for Histopathology Image Segmentation’, in 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI) (ISBI 2026), 2026, p. 4.
"""

import torch
import torch.nn as nn
from typing import Sequence
from networks.segmentation.ViT import ViT
from einops import rearrange


class MSConv(nn.Module):
    """multi-scale grouped convolution module"""

    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (1, 3, 5)) -> None:
        super(MSConv, self).__init__()
        conv2d_groups = dim // 2
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=conv2d_groups, bias=False)
            for kernel_size in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + sum([conv(x) for conv in self.dw_convs])


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and custom activation function"""

    def __init__(self, in_channels, out_channels, dropout_p, acti_func):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            acti_func(),
            nn.Dropout(dropout_p),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            MSConv(dim=out_channels),
            nn.BatchNorm2d(out_channels),
            acti_func()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p, acti_func):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p, acti_func)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear, acti_func):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p, acti_func)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_channels = self.params['in_channels']
        self.ft_chns = self.params['feature_channels']
        self.encoder_dropout = self.params['encoder_dropout']
        self.acti_func = self.params['activation_function']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_channels, self.ft_chns[0], self.encoder_dropout[0], self.acti_func)
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.encoder_dropout[1], self.acti_func)
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.encoder_dropout[2], self.acti_func)
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.encoder_dropout[3], self.acti_func)
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.encoder_dropout[4], self.acti_func)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.image_size = self.params['image_size']
        self.ft_chns = self.params['feature_channels']
        self.class_num = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.decoder_dropout = self.params['decoder_dropout']
        self.acti_func = self.params['activation_function']
        assert (len(self.ft_chns) == 5)

        self.vit_embedding_dim = self.params['vit_embedding_dim']
        self.vit_head_num = self.params['vit_head_num']
        self.vit_mlp_dim = self.params['vit_mlp_dim']
        self.vit_block_num = self.params['vit_block_num']
        self.vit_div_part = self.params['vit_div_part']
        self.use_vit = self.params['use_vit']

        if self.use_vit:
            # grouped residual Transformer module
            self.vit_patch_dim = self.image_size // 16
            self.vit = ViT(img_dim=self.vit_patch_dim, in_channels=self.ft_chns[-1] // self.vit_div_part,
                           embedding_dim=self.vit_embedding_dim, head_num=self.vit_head_num,
                           mlp_dim=self.vit_mlp_dim, block_num=self.vit_block_num,
                           patch_dim=1, classification=False)
            self.vit_conv = nn.Conv2d(self.vit_embedding_dim, self.ft_chns[-1] // self.vit_div_part, kernel_size=3, stride=1, padding=1)
            self.skip_scale = nn.Parameter(torch.ones(1))


        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3],
                           dropout_p=self.decoder_dropout[0], bilinear=self.bilinear, acti_func=self.acti_func)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2],
                           dropout_p=self.decoder_dropout[1], bilinear=self.bilinear, acti_func=self.acti_func)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1],
                           dropout_p=self.decoder_dropout[2], bilinear=self.bilinear, acti_func=self.acti_func)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0],
                           dropout_p=self.decoder_dropout[3], bilinear=self.bilinear, acti_func=self.acti_func)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.class_num, kernel_size=3, padding=1)


    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]  # torch.Size([8, 256, 16, 16])

        if self.use_vit:
            # ViT
            if self.vit_div_part == 1:
                x_vit = self.vit(x4)
                x_vit = rearrange(x_vit, "b (x y) c -> b c x y", x=self.vit_patch_dim, y=self.vit_patch_dim)
                x4 = self.vit_conv(x_vit)  # torch.Size([b, c, patch_dim, patch_dim])
            elif self.vit_div_part > 1:
                temp = []
                for one_part in torch.chunk(x4, self.vit_div_part, dim=1):
                    one_part_vit_ft = rearrange(self.vit(one_part), "b (x y) c -> b c x y", x=self.vit_patch_dim, y=self.vit_patch_dim)
                    one_part_vit_ft = self.vit_conv(one_part_vit_ft) + self.skip_scale * one_part  # torch.Size([b, c, patch_dim, patch_dim])
                    temp.append(one_part_vit_ft)
                x4 = torch.cat(temp, dim=1)
            else:
                assert (self.vit_div_part >= 1)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class LasagnaUNet(nn.Module):
    def __init__(self, image_size, in_channels, class_num):
        super(LasagnaUNet, self).__init__()

        params = {'image_size': image_size,
                  'in_channels': in_channels,
                  'feature_channels': [64, 128, 256, 512, 1024], # [16, 32, 64, 128, 256], [32, 64, 128, 256, 512], [64, 128, 256, 512, 1024]
                  'encoder_dropout': [0.05, 0.2, 0.2, 0.2, 0.2],
                  'decoder_dropout': [0.0, 0.0, 0.0, 0.0],
                  'class_num': class_num,
                  'bilinear': False,
                  'activation_function': nn.LeakyReLU,

                  'use_vit': True,
                  'vit_embedding_dim': 256,
                  'vit_head_num': 4,
                  'vit_mlp_dim': 256,
                  'vit_block_num': 8,
                  'vit_div_part': 4,  # If it is greater than 1, group attention is adopted

                  }

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

