#!/usr/bin/env python3
"""
Training script for VQWGAN on CIFAR-10 (stable SN-GAN style)

Key changes vs. original:
1) Hinge GAN loss + SpectralNorm (au lieu de WGAN+SN sans GP)
2) Un seul forward du générateur par itération (réutilisé pour D et G)
3) AdamW avec groupes (pas de weight decay sur bias/norm/quantize)
4) Option n_critic (1–2) après disc_start
5) AMP optionnelle + LPIPS optionnel (léger) pour meilleure perceptuelle

Sorties:
- checkpoints dans --output_dir (poids G/D + optims + args)
- échantillons de reconstructions périodiques
- PSNR sur un mini-batch de test
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from torch.nn.utils import spectral_norm

from models.vqwgan import VQWGAN  # doit fournir .encoder/.decoder/.quantize/.quant_conv/.post_quant_conv

# AMP / LPIPS (optionnels)
from torch.amp import autocast, GradScaler
try:
    import lpips
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False


# -----------------------------
# Discriminateur avec SN (CIFAR-10 32x32)
# -----------------------------
class ImprovedDiscriminator(nn.Module):
    """Patch/global discriminator simple avec Spectral Normalization"""
    def __init__(self, channels=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            # 32->16
            spectral_norm(nn.Conv2d(channels, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # 16->8
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # 8->4
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # 4->1
            spectral_norm(nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, x):
        # Retourne des logits (pas de sigmoïde)
        return self.main(x).view(-1)


# -----------------------------
# Args
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser('Stable VQWGAN CIFAR-10')

    # Modèle (VQWGAN)
    parser.add_argument('--vocab_size', default=4096, type=int)
    parser.add_argument('--z_channels', default=32, type=int)
    parser.add_argument('--ch', default=128, type=int)
    parser.add_argument('--beta', default=0.25, type=float)
    parser.add_argument('--quant_resi', default=0.5, type=float)
    parser.add_argument('--share_quant_resi', default=4, type=int)
    parser.add_argument('--v_patch_nums', default='1,2,4,8', type=str,
                        help='Patch nums pour 32x32 (ex: "1,2,4,8")')

    # Entraînement
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_g', default=1e-4, type=float)
    parser.add_argument('--lr_d', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    # GAN
    parser.add_argument('--loss', default='hinge', choices=['hinge', 'wgan'])
    parser.add_argument('--disc_start', default=5000, type=int, help='Activer l’adversarial après N steps')
    parser.add_argument('--disc_weight', default=0.2, type=float, help='Poids du loss adversarial côté G')
    parser.add_argument('--disc_factor', default=1.0, type=float, help='Multiplicateur du loss D')
    parser.add_argument('--n_critic', default=1, type=int, help='Updates de D par update de G (après disc_start)')
    parser.add_argument('--adaptive_weight', action='store_true', help='Pondération adaptative (grad-ratio)')

    # Options
    parser.add_argument('--use_lpips', action='store_true', help='Ajoute LPIPS (alex) x0.1 à la recon')
    parser.add_argument('--amp', action='store_true', help='AMP autocast + GradScaler')

    # Divers
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--output_dir', default='./output/vqwgan_stable', type=str)
    parser.add_argument('--log_freq', default=50, type=int)
    parser.add_argument('--save_freq', default=10, type=int)

    return parser.parse_args()


# -----------------------------
# Pondération adaptative (optionnelle)
# -----------------------------
def calculate_adaptive_weight(rec_loss, g_loss, last_layer):
    rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight


# -----------------------------
# Optim param groups (AdamW)
# -----------------------------
def split_params_adamw(model, weight_decay):
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name = n.lower()
        if name.endswith('bias') or 'norm' in name or 'bn' in name or 'quant' in name:
            nodecay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Parse v_patch_nums
    v_patch_nums = tuple(int(x) for x in args.v_patch_nums.split(','))
    os.makedirs(args.output_dir, exist_ok=True)

    print("===============================================")
    print("  VQWGAN CIFAR-10 (Stable SN-GAN configuration) ")
    print("===============================================")
    print(f"Device: {device} | AMP: {args.amp} | LPIPS: {args.use_lpips and HAS_LPIPS}")
    print(f"Loss: {args.loss} | n_critic: {args.n_critic} | disc_start: {args.disc_start}")
    print(f"vocab_size: {args.vocab_size} | v_patch_nums: {v_patch_nums}")

    # Modèle
    generator = VQWGAN(
        vocab_size=args.vocab_size,
        z_channels=args.z_channels,
        ch=args.ch,
        beta=args.beta,
        quant_resi=args.quant_resi,
        share_quant_resi=args.share_quant_resi,
        v_patch_nums=v_patch_nums,
        test_mode=False,  # training!
    ).to(device)

    discriminator = ImprovedDiscriminator(channels=3, ndf=64).to(device)

    # Optimizers (AdamW + param groups)
    optimizer_g = torch.optim.AdamW(
        split_params_adamw(generator, args.weight_decay),
        lr=args.lr_g, betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args.lr_d, betas=(0.5, 0.999)
    )

    # LR schedulers (cosine per-step)
    # On steppe à chaque batch après step optimizer
    # T_max = total_steps
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=total_steps)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=total_steps)

    # LPIPS (optionnel)
    lpips_alex = lpips.LPIPS(net='alex').to(device).eval() if (HAS_LPIPS and args.use_lpips) else None

    # AMP
    scaler = GradScaler('cuda', enabled=args.amp)

    print("\n🏋️  Starting training…")
    global_step = 0

    for epoch in range(args.epochs):
        generator.train(); discriminator.train()
        epoch_stats = {'rec': 0.0, 'vq': 0.0, 'g': 0.0, 'd': 0.0}

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)

            # -------- Forward unique du générateur --------
            with autocast('cuda', enabled=args.amp):
                rec_images, usages, vq_loss = generator(images, ret_usages=True)

                # Recon loss
                rec_loss = F.l1_loss(rec_images, images) + 0.5 * F.mse_loss(rec_images, images)

                # LPIPS optionnel (entrées en [0,1])
                if lpips_alex is not None:
                    rec01 = rec_images.add(1).mul(0.5).clamp(0, 1)
                    img01 = images.add(1).mul(0.5).clamp(0, 1)
                    perc = lpips_alex(rec01, img01).mean()
                    rec_loss = rec_loss + 0.1 * perc

                # Guard VQ explosion
                if vq_loss.detach().item() > 100.0:
                    vq_loss = torch.clamp(vq_loss, 0.0, 100.0)

            # -------- D-step (n_critic après disc_start) --------
            do_gan = (global_step >= args.disc_start)
            if do_gan:
                optimizer_d.zero_grad(set_to_none=True)
                d_loss_total = 0.0

                for _ in range(max(1, args.n_critic)):
                    with autocast(device_type=device.type, enabled=args.amp):

                        # réutilise rec_images.detach() du forward unique
                        real_out = discriminator(images)
                        fake_out = discriminator(rec_images.detach())

                        if args.loss == 'hinge':
                            d_loss = F.relu(1.0 - real_out).mean() + F.relu(1.0 + fake_out).mean()
                        else:  # wgan
                            d_loss = -real_out.mean() + fake_out.mean()

                        d_loss = d_loss * args.disc_factor

                    # accumulation : on normalise pour garder la même échelle de gradient
                    scaler.scale(d_loss / max(1, args.n_critic)).backward()
                    d_loss_total += d_loss.detach().item()

                # un seul step / update / scheduler après accumulation
                scaler.step(optimizer_d)
                # ne pas appeler scaler.update() ici si tu enchaînes avec le G-step via le même scaler ?
                # -> si tu utilises UN SEUL scaler pour D et G, tu peux appeler update APRÈS les 2 steps.
                # sinon, appelle update ici et n'en remets pas un plus tard.
                # version simple: on fait update ici et on laissera le G-step aussi faire un update (c'est OK).
                scaler.update()
                scheduler_d.step()

                d_loss_log = d_loss_total  # pour les stats
            else:
                d_loss_log = 0.0
                real_out = fake_out = torch.tensor(0.0, device=device)

            # -------- G-step --------
            optimizer_g.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=args.amp):
                if do_gan:
                    fake_out_g = discriminator(rec_images)
                    if args.loss == 'hinge':
                        g_adv = -fake_out_g.mean()
                    else:  # wgan
                        g_adv = -fake_out_g.mean()

                    # Pondération adaptative éventuelle (peu fréquente)
                    if args.adaptive_weight and (global_step % 200 == 0):
                        try:
                            last_layer = generator.decoder.conv_out.weight
                            aw = calculate_adaptive_weight(rec_loss, g_adv, last_layer)
                            disc_weight = torch.clamp(aw * args.disc_weight, 0, 1e4)
                        except Exception:
                            disc_weight = torch.tensor(args.disc_weight, device=device)
                    else:
                        disc_weight = torch.tensor(args.disc_weight, device=device)

                    total_loss = rec_loss + vq_loss + disc_weight * g_adv
                else:
                    g_adv = torch.tensor(0.0, device=device)
                    total_loss = rec_loss + vq_loss

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            scaler.step(optimizer_g)
            scaler.update()
            scheduler_g.step()

            # -------- Stats / logs --------
            epoch_stats['rec'] += float(rec_loss.detach().item())
            epoch_stats['vq'] += float(vq_loss.detach().item())
            if do_gan:
                epoch_stats['g'] += float(g_adv.detach().item())
                epoch_stats['d'] += float(d_loss.detach().item())

            if global_step % args.log_freq == 0:
                mode = "warmup" if not do_gan else "GAN"
                rscore = real_out.mean().item() if do_gan else 0.0
                fscore = fake_out.mean().item() if do_gan else 0.0
                print(f"[{mode}] Ep[{epoch}/{args.epochs}] Step[{global_step}] "
                      f"rec:{epoch_stats['rec']/(batch_idx+1):.4f} "
                      f"vq:{epoch_stats['vq']/(batch_idx+1):.4f} "
                      f"g:{(epoch_stats['g']/(batch_idx+1) if do_gan else 0.0):.4f} "
                      f"d:{(epoch_stats['d']/(batch_idx+1) if do_gan else 0.0):.4f} "
                      f"| D(real):{rscore:.3f} D(fake):{fscore:.3f}")

            global_step += 1

        # Fin d’epoch
        n_batches = len(train_loader)
        print(f"✓ Epoch {epoch} avg - "
              f"rec: {epoch_stats['rec']/n_batches:.4f} "
              f"vq: {epoch_stats['vq']/n_batches:.4f} "
              f"g: {epoch_stats['g']/n_batches:.4f} "
              f"d: {epoch_stats['d']/n_batches:.4f}")

        # Save + petite éval de reconstruction
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.output_dir, f'vqwgan_epoch{epoch}.pth')
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'args': args,
            }, save_path)
            print(f"   Saved: {save_path}")

            # Visualisation / PSNR
            generator.eval()
            with torch.no_grad():
                # mini-batch fixe
                sample_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                                           num_workers=0, drop_last=True)
                test_imgs, _ = next(iter(sample_loader))
                test_imgs = test_imgs.to(device)
                test_rec, _, _ = generator(test_imgs)

                # PSNR (images en [-1,1], max=4)
                mse = F.mse_loss(test_imgs, test_rec).item()
                psnr = 10 * torch.log10(torch.tensor(4.0) / (mse + 1e-10)).item()
                print(f"   Test MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

                # Sauvegarde d’un comparatif
                try:
                    import torchvision.utils as vutils
                    comparison = torch.cat([test_imgs[:8], test_rec[:8]], dim=0)
                    img_path = os.path.join(args.output_dir, f'samples_epoch{epoch}.png')
                    vutils.save_image(comparison, img_path, nrow=8, normalize=True, value_range=(-1, 1))
                    print(f"   Saved samples: {img_path}")
                except Exception as e:
                    print(f"   Could not save images: {e}")

            generator.train()

    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
