import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        ######
        # номера слоев считать по уровням сверху вниз, слева направо
        ######

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            # 1, conv+bn+relu
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 2, conv+bn+relu
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 3, pool
        self.pool0 = nn.MaxPool2d(2, 2)  # 256 -> 128

        self.enc_conv1 = nn.Sequential(
            # 4, conv+bn+relu
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 5, conv+bn+relu
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 6, pool
        self.pool1 = nn.MaxPool2d(2, 2)  # 128 -> 64

        self.enc_conv2 = nn.Sequential(
            # 7, conv+bn+relu
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 8, conv+bn+relu
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 9, pool
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 -> 32

        self.enc_conv3 = nn.Sequential(
            # 10, conv+bn+relu
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 11, conv+bn+relu
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # 12, pool
        self.pool3 = nn.MaxPool2d(2, 2)  # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            # 13, conv+bn+relu
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            # 14, conv+bn+relu
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # decoder (upsampling)

        # 15, upsample
        self.upsample0 = nn.Upsample(32)  # 16 -> 32

        self.dec_conv0 = nn.Sequential(
            # 16, conv+bn+relu
            nn.Conv2d(in_channels=1024 + 512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 17, conv+bn+relu
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # 18, upsample
        self.upsample1 = nn.Upsample(64)  # 32 -> 64

        self.dec_conv1 = nn.Sequential(
            # 19, conv+bn+relu
            nn.Conv2d(in_channels=512 + 256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 20, conv+bn+relu
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 21, upsample
        self.upsample2 = nn.Upsample(128)  # 64 -> 128

        self.dec_conv2 = nn.Sequential(
            # 22, conv+bn+relu
            nn.Conv2d(in_channels=256 + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 23, conv+bn+relu
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 24, upsample
        self.upsample3 = nn.Upsample(256)  # 128 -> 256

        self.dec_conv3 = nn.Sequential(
            # 25, conv+bn+relu
            nn.Conv2d(in_channels=128 + 64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 26, conv+bn+relu
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 27, conv+bn
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1)
            #         nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # encoder
        ##enc0
        e0 = self.enc_conv0(x)

        ##enc1
        e1 = self.enc_conv1(self.pool0(e0))

        ##enc2
        e2 = self.enc_conv2(self.pool1(e1))

        ##enc3
        e3 = self.enc_conv3(self.pool2(e2))

        # bottleneck
        bn = self.bottleneck_conv(self.pool3(e3))

        # decoder
        ##dec0
        d0 = self.upsample0(bn)
        d0 = torch.cat((d0, e3), dim=1)
        d0 = self.dec_conv0(d0)

        ##dec1
        d1 = self.upsample1(d0)
        d1 = torch.cat((d1, e2), dim=1)
        d1 = self.dec_conv1(d1)

        # dec2
        d2 = self.upsample2(d1)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.dec_conv2(d2)

        # dec2
        d3 = self.upsample3(d2)
        d3 = torch.cat((d3, e0), dim=1)
        d3 = self.dec_conv3(d3)

        return d3  # no activation
