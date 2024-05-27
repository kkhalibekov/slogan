import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = self.layer(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.layer(in_channels)

    def layer(self, ins):
        outs = ins
        return nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outs),
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            self.layer_conv(16),
            self.layer_conv(32),
            self.layer_conv(64),
            self.layer_conv(128),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(6)]
        )
        self.decoder = nn.Sequential(
            self.layer_deconv(256),
            self.layer_deconv(128),
            self.layer_deconv(64),
            self.layer_deconv(32),
            nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
        )

    def layer_conv(self, ins):
        outs = ins * 2
        return nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(outs),
            nn.ReLU(inplace=True),
        )

    def layer_deconv(self, ins):
        outs = ins // 2
        return nn.Sequential(
            nn.ConvTranspose2d(
                ins, outs, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(outs),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, style_vec):
        style_vec = style_vec.view(style_vec.size(0), style_vec.size(1), 1, 1)
        style_vec = style_vec.expand(
            style_vec.size(0), style_vec.size(1), x.size(2), x.size(3)
        )
        x = torch.cat((x, style_vec), 1)
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x


class StyleBank(nn.Module):
    def __init__(self, num_styles, style_dim):
        super().__init__()
        self.style_bank = nn.Parameter(torch.randn(num_styles, style_dim))

    def forward(self, style_ids):
        return self.style_bank[style_ids]


class CharDiscriminator(nn.Module):
    """Separated Character Discriminator"""

    def __init__(self, num_chars):
        super().__init__()
        self.conv = nn.Sequential(
            self.layer_conv_pool(3, 16),
            self.layer_conv_pool(16, 64),
            self.layer_conv_pool(64, 128),
            self.layer_conv_pool(128, 128),
            self.layer_conv_pool(128, 192, pool_stride=1),
            self.layer_conv(192, 256),
            self.layer_conv(256, 256),
            nn.Flatten(),
        )
        self.fc_adv = nn.Linear(256 * 2 * 25, 1)
        self.fc_content = nn.Linear(256 * 2 * 25, num_chars)

    def layer_conv(self, ins, outs):
        return nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(outs),
            nn.PReLU(),
        )

    def layer_conv_pool(self, ins, outs, pool_stride=2):
        return nn.Sequential(
            self.layer_conv(ins, outs),
            nn.AvgPool2d(2, pool_stride),
        )

    def forward(self, x):
        x = self.conv(x)
        adv_out = self.fc_adv(x)
        content_out = self.fc_content(x)
        return adv_out, content_out


class JoinDiscriminator(nn.Module):
    """Cursive Join Discriminator"""

    def __init__(self, num_styles):
        super().__init__()
        self.conv = nn.Sequential(
            self.layer_conv_pool(3, 16),
            self.layer_conv_pool(16, 64),
            self.layer_conv_pool(64, 128),
            self.layer_conv_pool(128, 128),
        )
        self.adv = nn.Sequential(
            self.layer_conv_pool(128, 64),
            self.layer_conv(64, 16),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        self.style = nn.Sequential(
            self.layer_conv_pool(128, 192),
            self.layer_conv_pool(192, 256),
            nn.Conv2d(256, num_styles, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(1, 7),
        )

    def layer_conv(self, ins, outs):
        return nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(outs),
            nn.PReLU(),
        )

    def layer_conv_pool(self, ins, outs):
        return nn.Sequential(
            self.layer_conv(ins, outs),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        adv_out = self.adv(x)
        style_out = self.style(x)
        return adv_out, style_out
