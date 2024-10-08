import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import lightning as L

from model import (
    Generator,
    StyleBank,
    Discriminator,
)


class SLOGAN(L.LightningModule):
    def __init__(self, num_chars, num_styles, latent_dim):
        super().__init__()

        self.generator = Generator(latent_dim)
        self.style_bank = StyleBank(num_styles, latent_dim)
        self.discriminator = Discriminator(num_chars, num_styles)

        self.automatic_optimization = False

    def forward(self, x, writer_id):
        style_vec = self.style_bank(writer_id)
        return self.generator(x, style_vec)

    def training_step(self, batch, batch_idx):
        print_img, print_label, real_img, real_label, writer_id = batch
        d_opt, g_opt, sb_opt = self.optimizers()
        d_sch, g_sch, sb_sch = self.lr_schedulers()

        fake_img = self(print_img, writer_id)

        ####################
        # 0. Discriminator step
        ####################

        fake_img_detach = fake_img.detach()

        fake_char_adv, fake_char_content, fake_join_adv, fake_join_style = (
            self.discriminator(fake_img_detach)
        )

        real_char_adv, real_char_content, real_join_adv, real_join_style = (
            self.discriminator(real_img)
        )

        # TODO l_char_content = loss_char_content
        # TODO l_char_adv = loss_char_adv
        loss_char = l_char_adv + l_char_content

        # TODO l_join_adv = loss_join_adv(fake_join_adv, real_join_adv)
        # TODO l_join_style = loss_join_style(writer_id, real_img)
        loss_join = l_join_adv + l_join_style

        d_opt.zero_grad()
        self.manual_backward(loss_char, retain_graph=True)
        self.manual_backward(loss_join)
        d_opt.step()
        d_sch.step()

        ####################
        # 1. Generator step
        ####################

        (fake_char_adv, fake_char_content, fake_join_adv, fake_join_style) = (
            self.discriminator(fake_img)
        )

        g_loss = (
            # TODO loss_char_adv
            # + TODO loss_char_content
            # + TODO loss_join_adv
        )
        g_sb_loss = (
            # TODO loss_join_style(writer_id, fake_img)
            # + TODO loss_idt(real_img, writer_id)
        )

        g_opt.zero_grad()
        sb_opt.zero_grad()
        self.manual_backward(g_loss, retain_graph=True)
        self.manual_backward(g_sb_loss)
        g_opt.step()
        sb_opt.step()
        g_sch.step()
        sb_sch.step()

        ####################

        self.log_dict(
            {
                "loss_char": loss_char,
                "loss_join": loss_join,
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        lr_init = 1e-4
        lr_target = 1e-5
        num_iters = 300_000
        betas = (0.5, 0.999)

        d_opt = Adam(
            self.discriminator.parameters(),
            lr=lr_init,
            betas=betas,
        )
        g_opt = Adam(
            self.generator.parameters(),
            lr=lr_init,
            betas=betas,
        )
        sb_opt = Adam(
            self.style_bank.parameters(),
            lr=lr_init,
            betas=betas,
        )

        d_sch = LinearLR(d_opt, 1, lr_target / lr_init, num_iters)
        g_sch = LinearLR(g_opt, 1, lr_target / lr_init, num_iters)
        sb_sch = LinearLR(sb_opt, 1, lr_target / lr_init, num_iters)

        return [d_opt, g_opt, sb_opt], [
            {
                "scheduler": d_sch,
                "name": "d_sch",
            },
            {
                "scheduler": g_sch,
                "name": "g_sch",
            },
            {
                "scheduler": sb_sch,
                "name": "sb_sch",
            },
        ]
