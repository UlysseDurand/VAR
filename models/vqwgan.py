"""
VQWGAN: Vector Quantized Wasserstein GAN that we added 
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_vae import Decoder, Encoder
from .quant import VectorQuantizer2


class WassersteinDiscriminator(nn.Module):
    """
    Wasserstein discriminator (critic) with spectral normalization
    Outputs a real-valued score instead of a probability
    """
    def __init__(
        self,
        in_channels=3,
        ch=64,
        num_layers=3,
        use_spectral_norm=False,
    ):
        super().__init__()
        
        norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        layers = []
        current_ch = in_channels
        
        for i in range(num_layers):
            out_ch = ch * (2 ** i)
            layers.extend([
                norm_layer(nn.Conv2d(current_ch, out_ch, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            current_ch = out_ch
        
        #conv layers without downsampling
        layers.extend([
            norm_layer(nn.Conv2d(current_ch, current_ch, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        
        # Final layer to output single value per spatial location
        layers.append(norm_layer(nn.Conv2d(current_ch, 1, kernel_size=3, stride=1, padding=1)))
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Critic score [B, 1, H', W'] where H', W' are reduced dimensions
        """
        return self.main(x)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for local realism
    Operates on image patches for texture quality
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        
        # Use spectral normalization for Lipschitz constraint
        norm_layer = nn.utils.spectral_norm
        
        sequence = [
            norm_layer(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        sequence += [norm_layer(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))]
        
        self.main = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.main(x)


class VQWGAN(nn.Module):
    """
    Vector Quantized Wasserstein GAN
    """
    def __init__(
        self,
        vocab_size=4096,
        z_channels=32,
        ch=128,
        dropout=0.0,
        beta=0.25,              # commitment loss weight
        using_znorm=False,
        quant_conv_ks=3,
        quant_resi=0.5,
        share_quant_resi=4,
        default_qresi_counts=0,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        # Wasserstein GAN specific parameters
        disc_ch=64,             # discriminator channels
        disc_num_layers=3,      # number of discriminator layers
        disc_start=10000,       # iteration to start discriminator training
        disc_weight=0.5,        # weight for adversarial loss
        use_patch_disc=True,    # use patch discriminator
        gp_weight=10.0,         # gradient penalty weight
        perceptual_weight=1.0,  # perceptual loss weight
        test_mode=True,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 2, 4), num_res_blocks=2,
            using_sa=True, using_mid_sa=True,
        )
        ddconfig.pop('double_z', None)
        
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult']) - 1)
        
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums,
            quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        
        self.quant_conv = nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        self.post_quant_conv = nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        
        # Initialize the quantizer codebook
        self.quantize.eini(0.02)  # Small initialization for better stability
        
        # Wasserstein discriminator components
        self.discriminator = WassersteinDiscriminator(
            in_channels=3,
            ch=disc_ch,
            num_layers=disc_num_layers,
            use_spectral_norm=False,
        )
        
        if use_patch_disc:
            self.patch_discriminator = PatchDiscriminator(
                in_channels=3,
                ndf=disc_ch,
                n_layers=disc_num_layers,
            )
        else:
            self.patch_discriminator = None
        
        # Wasserstein GAN parameters
        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.gp_weight = gp_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_iter_start = 0
        
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    def encode(self, x):
        """Encode image to latent features"""
        return self.quant_conv(self.encoder(x))
    
    def decode(self, f_hat):
        """Decode latent features to image"""
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
    def forward(self, inp, ret_usages=False):
        """
        Forward pass for training
        Returns: reconstruction, usages, vq_loss
        """
        f = self.encode(inp)
        
        # CRITICAL: Clamp latents to prevent explosion
        # This prevents the encoder from producing extreme values under adversarial pressure
        f = torch.clamp(f, -10.0, 10.0)
        
        f_hat, usages, vq_loss = self.quantize(f, ret_usages=ret_usages)
        
        # Additional safety: clamp vq_loss to prevent gradient explosion
        if isinstance(vq_loss, torch.Tensor):
            vq_loss = torch.clamp(vq_loss, 0.0, 100.0)
        
        rec = self.decode(f_hat)
        return rec, usages, vq_loss
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """
        Compute gradient penalty for WGAN-GP
        Enforces Lipschitz constraint on discriminator
        """
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)
        
        # Interpolate between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # Get discriminator output
        d_interpolates = self.discriminator(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def discriminator_loss(self, real_imgs, fake_imgs):
        """
        Compute Wasserstein discriminator loss with gradient penalty
        Discriminator maximizes: E[D(real)] - E[D(fake)] - λ·GP
        """
        # Real images
        real_validity = self.discriminator(real_imgs)
        real_loss = -real_validity.mean()
        
        # Fake images
        fake_validity = self.discriminator(fake_imgs.detach())
        fake_loss = fake_validity.mean()
        
        # Gradient penalty
        gp = self.compute_gradient_penalty(real_imgs, fake_imgs.detach())
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss + self.gp_weight * gp
        
        # Optional: patch discriminator
        if self.patch_discriminator is not None:
            real_patch = self.patch_discriminator(real_imgs)
            fake_patch = self.patch_discriminator(fake_imgs.detach())
            patch_loss = -real_patch.mean() + fake_patch.mean()
            d_loss = d_loss + patch_loss
        
        return d_loss, {
            'disc_real': -real_loss.item(),
            'disc_fake': fake_loss.item(),
            'gradient_penalty': gp.item(),
        }
    
    def generator_loss(self, fake_imgs):
        """
        Compute generator (VAE) adversarial loss
        Generator minimizes: -E[D(fake)]
        """
        fake_validity = self.discriminator(fake_imgs)
        g_loss = -fake_validity.mean()
        
        if self.patch_discriminator is not None:
            fake_patch = self.patch_discriminator(fake_imgs)
            g_loss = g_loss - fake_patch.mean()
        
        return g_loss
    
    def compute_loss(self, inp, global_step):
        """
        Compute total training loss
        """
        # Forward pass
        rec, usages, vq_loss = self.forward(inp, ret_usages=True)
        
        # Reconstruction loss (L1 + L2 for stability)
        rec_loss = F.l1_loss(rec, inp) + 0.5 * F.mse_loss(rec, inp)
        
        # Adversarial loss (only after disc_start iterations)
        if global_step >= self.disc_start:
            g_loss = self.generator_loss(rec)
            # Gradually ramp up adversarial loss
            adv_weight = min(1.0, (global_step - self.disc_start) / 5000.0) * self.disc_weight
            total_loss = rec_loss + vq_loss + adv_weight * g_loss
        else:
            g_loss = torch.tensor(0.0, device=inp.device)
            total_loss = rec_loss + vq_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'rec_loss': rec_loss.item(),
            'vq_loss': vq_loss.item(),
            'g_loss': g_loss.item() if isinstance(g_loss, torch.Tensor) else g_loss,
            'usages': usages,
        }
    
    # methods copied from VQVAE in this repo
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decode(f_hat)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums=None) -> List[torch.LongTensor]:
        f = self.encode(inp_img_no_grad)
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False):
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False):
        if last_one:
            f_hat = self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True)
            return self.decode(f_hat)
        else:
            f_hats = self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)
            return [self.decode(f_hat) for f_hat in f_hats]
    
    def img_to_reconstructed_img(self, x, v_patch_nums=None, last_one=False):
        f = self.encode(x)
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decode(ls_f_hat_BChw[-1])
        else:
            return [self.decode(f_hat) for f_hat in ls_f_hat_BChw]
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict:
            if state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
                state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
