import torch
from torch import nn


class StyleBank(nn.Module):
    def __init__(self, num_styles, latent_dim):
        super().__init__()
        self.style_bank = nn.Parameter(torch.randn(num_styles, latent_dim))

    def forward(self, writer_id):
        return self.style_bank[writer_id]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ) + (
            self.layer_conv(16)
            + self.layer_conv(32)
            + self.layer_conv(64)
            + self.layer_conv(128)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(3)]
        )
        self.residual_blocks_style = nn.Sequential(
            *[ResidualBlock(256 + latent_dim) for _ in range(3)]
        )
        self.deconv = (
            self.layer_deconv(256 + latent_dim, 128)
            + self.layer_deconv(128)
            + self.layer_deconv(64)
            + self.layer_deconv(32)
            + nn.Sequential(
                nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2),
                nn.Tanh(),
            )
        )

    def forward(self, x, style_vec):
        x = self.conv(x)
        x = self.residual_blocks(x)

        # concatenation the style vector with the output feature maps of the 3rd residual block
        # x (b, c, h, w)
        # style_vec (b, d)
        style_vec = style_vec.unsqueeze(-1).unsqueeze(-1)  # (b, d, 1, 1)
        style_vec = style_vec.repeat(
            1, 1, x.size(2), x.size(3)
        )  # (b, d, h, w)
        x = torch.cat((x, style_vec), 1)  # (b, c + d, h, w)

        x = self.residual_blocks_style(x)
        x = self.deconv(x)
        return x

    def layer_conv(self, ins):
        outs = ins * 2
        return nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(outs),
            nn.ReLU(inplace=True),
        )

    def layer_deconv(self, ins, outs=None):
        if outs is None:
            outs = ins // 2
        return nn.Sequential(
            nn.ConvTranspose2d(
                ins, outs, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(outs),
            nn.ReLU(inplace=True),
        )

############################


def discr_layer_conv(ins, outs):
    return nn.Sequential(
        nn.Conv2d(ins, outs, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(outs),
        nn.PReLU(),
    )


def discr_layer_conv_pool(ins, outs, pool_size=2):
    return discr_layer_conv(ins, outs) + nn.Sequential(
        nn.AvgPool2d(pool_size),
    )


class DiscrSharedLayers(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = (
            discr_layer_conv_pool(3, 16)
            + discr_layer_conv_pool(16, 64)
            + discr_layer_conv_pool(64, 128)
            + discr_layer_conv_pool(128, 128)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DiscrChar(nn.Module):
    """Separated Character Discriminator"""

    def __init__(self, num_chars):
        super().__init__()

        self.conv = (
            discr_layer_conv_pool(128, 192, pool_size=(2, 1))
            + discr_layer_conv(192, 256)
            + discr_layer_conv(256, 256)
        )

        self.attention_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 25, 256),
        )
        self.gru = nn.GRU(256, 256, batch_first=True)

        self.adv = nn.Linear(256, 1)
        self.content = nn.Linear(256, num_chars)

    def forward(self, x):
        x = self.conv(x)

        x = self.attention_fc(x)
        x = x.unsqueeze(1)
        x, _ = self.gru(x)
        x = x.squeeze(1)

        adv_out = self.adv(x)
        content_out = self.content(x)
        return adv_out, content_out


class DiscrJoin(nn.Module):
    """Cursive Join Discriminator"""

    def __init__(self, num_styles):
        super().__init__()

        self.adv = nn.Sequential(
            discr_layer_conv_pool(128, 64),
            discr_layer_conv(64, 16),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        self.style = nn.Sequential(
            discr_layer_conv_pool(128, 192),
            discr_layer_conv_pool(192, 256),
            nn.Conv2d(256, num_styles, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(1, 7),
        )

    def forward(self, x):
        adv_out = self.adv(x)
        style_out = self.style(x)
        return adv_out, style_out


class Discriminator(nn.Module):
    def __init__(self, num_chars, num_styles) -> None:
        super().__init__()

        self.shared_conv = DiscrSharedLayers()
        self.char_discr = DiscrChar(num_chars)
        self.join_discr = DiscrJoin(num_styles)

    def forward(self, x):
        x = self.shared_conv(x)
        char_adv, char_content = self.char_discr(x)
        join_adv, join_style = self.join_discr(x)
        return char_adv, char_content, join_adv, join_style
