import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, concat=False):
        super(UNet, self).__init__()
        self.concat = concat
        self.expansion = 2 if concat else 1
        # Contracting Path
        self.encoder1 = self.contract_block(in_channels, 64)
        self.encoder2 = self.contract_block(64, 128, downsample=True)
        self.encoder3 = self.contract_block(128, 256, downsample=True)
        self.encoder4 = self.contract_block(256, 512, downsample=True)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(512),
        )

        # Expanding Path
        self.decoder3 = self.expand_block(512*self.expansion, 512, 256)
        self.decoder2 = self.expand_block(256*self.expansion, 256, 128)
        self.decoder1 = self.expand_block(128*self.expansion, 128, 64)
        self.out = nn.Conv2d(64, out_channels, 1, 1, 0, 1)

        if not concat:
            self.map3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True)
            self.map2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
            self.map1 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=True)
            # self.map0 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        # x = self.encoder0(x)
        # Contracting Path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Expanding Path
        if self.concat:
            dec3 = self.decoder3(torch.cat((bottleneck, enc4), 1))
            dec2 = self.decoder2(torch.cat((dec3, enc3), 1))
            dec1 = self.decoder1(torch.cat((dec2, enc2), 1))
        else:
            dec3 = self.decoder3(F.relu(bottleneck) + self.map3(enc4))
            dec2 = self.decoder2(F.relu(dec3) + self.map2(enc3))
            dec1 = self.decoder1(F.relu(dec2) + self.map1(enc2))

        return self.out(F.relu(dec1))

    def contract_block(self, in_channels, out_channels, kernel_size=3, downsample=False):
        if downsample:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        return block

    def expand_block(self, in_channels, mid_channel, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channel, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        )
        return block

    def pool(self, x):
        return F.max_pool2d(x, 2)

