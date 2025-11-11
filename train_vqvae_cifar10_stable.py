#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stable VQ-VAE (no GAN) training on CIFAR-10.
- Robust codebook init on first batch (fixed)
- Latent normalization + clamp
- Distances guarded (nan_to_num)
- EMA codebook with clamps
- AMP warmup (bf16 or fp16) + grad clipping
- Cosine LR schedule with warmup
- Recon dumps + codebook usage/perplexity

Run example at the bottom.
"""

import os
import math
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import save_image


def psnr_from_mse(mse, peak=2.0):
    return 10.0 * torch.log10(torch.tensor(peak * peak) / (mse + 1e-12))


# =========================
# Encoder / Decoder
# =========================

class Encoder(nn.Module):
    def __init__(self, in_ch=3, z_ch=128, ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1),   # 32 -> 16
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1),  # 16 -> 8
            nn.ReLU(True),
            nn.Conv2d(ch * 2, z_ch, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)  # [B, z_ch, 8, 8]


class Decoder(nn.Module):
    def __init__(self, out_ch=3, z_ch=128, ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_ch, ch * 2, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1),  # 8 -> 16
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch, out_ch, 4, 2, 1),  # 16 -> 32
            nn.Tanh(),  # outputs in [-1, 1]
        )

    def forward(self, z_q):
        return self.net(z_q)  # [B, 3, 32, 32]


# =========================
# Vector Quantizer (ST + EMA)
# =========================

class VectorQuantizer(nn.Module):
    """
    VQ with straight-through.
    If ema=True: EMA updates (VQ-VAE v2 style). Commitment only in that case.
    """
    def __init__(self, n_embed=512, embed_dim=128, beta=0.25, ema=False, ema_decay=0.99, eps=1e-5):
        super().__init__()
        self.n_embed = int(n_embed)
        self.embed_dim = int(embed_dim)
        self.beta = float(beta)
        self.ema = bool(ema)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)

        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        # small init; we’ll reinit from data at first batch anyway
        nn.init.normal_(self.embedding.weight, std=0.02)

        if self.ema:
            self.register_buffer("ema_cluster_size", torch.zeros(self.n_embed))
            self.register_buffer("ema_embed", self.embedding.weight.data.clone())

        # one-time data-driven init flag
        self._inited = False

    @torch.no_grad()
    def _first_batch_init(self, flat_inputs: torch.Tensor):
        """
        Data-driven init of codebook: pick n_embed diverse samples (kmeans++-like lite).
        """
        N = flat_inputs.shape[0]
        if N < self.n_embed:
            # pad repeats if batch too small
            reps = (self.n_embed + N - 1) // N
            cand = flat_inputs.repeat((reps, 1))[:self.n_embed]
            self.embedding.weight.data.copy_(cand)
            return

        # sample more points then choose spread-out centers
        m = min(self.n_embed * 20, N)
        cand = flat_inputs[torch.randperm(N, device=flat_inputs.device)[:m]]  # [m, C]

        # pick first center correctly as [1, C]
        first_idx = torch.randint(0, m, (), device=cand.device).item()
        first = cand[first_idx].unsqueeze(0)  # [1, C]
        centers = [first]
        d2 = torch.cdist(cand, first).pow(2).squeeze(-1)  # [m]

        for _ in range(1, self.n_embed):
            probs = (d2 + 1e-8) / (d2.sum() + 1e-8)
            new_idx = torch.multinomial(probs, 1).item()
            new_c = cand[new_idx].unsqueeze(0)  # [1, C]
            centers.append(new_c)
            d2 = torch.minimum(d2, torch.cdist(cand, new_c).pow(2).squeeze(-1))

        centers = torch.cat(centers, dim=0)  # [n_embed, C]
        self.embedding.weight.data.copy_(centers)

    @torch.no_grad()
    def _ema_update(self, flat_inputs, encodings):
        # encodings: [B*H*W, n_embed] one-hot
        cs = encodings.sum(0)  # [n_embed]
        self.ema_cluster_size.mul_(self.ema_decay).add_(cs, alpha=1.0 - self.ema_decay)

        embed_sum = flat_inputs.t() @ encodings  # [C, n_embed]
        self.ema_embed.mul_(self.ema_decay).add_(embed_sum.t(), alpha=1.0 - self.ema_decay)

        n = self.ema_cluster_size.sum().clamp_min(1.0)
        cluster_size = (self.ema_cluster_size + self.eps)
        cluster_size = (cluster_size / (n + self.n_embed * self.eps)) * n
        cluster_size = cluster_size.clamp_min(1.0)

        embed_normalized = self.ema_embed / cluster_size.unsqueeze(1)
        embed_normalized = torch.nan_to_num(embed_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        self.embedding.weight.data.copy_(embed_normalized)

    def forward(self, z):
        """
        z: [B, C, H, W]
        returns:
          z_q_st: quantized latent (ST)
          vq_loss: scalar
          idx: [B, H*W] token indices
        """
        B, C, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        flat = z_perm.view(-1, C)                    # [B*H*W, C]

        # one-time data-driven init
        if self.training and not self._inited:
            self._first_batch_init(flat)
            self._inited = True

        e_w = self.embedding.weight  # [n_embed, C]

        # L2 distances, guarded
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + e_w.pow(2).sum(dim=1, keepdim=True).t()
            - 2.0 * (flat @ e_w.t())
        )
        dist = torch.nan_to_num(dist, nan=1e4, posinf=1e4, neginf=1e4)

        idx = torch.argmin(dist, dim=1)  # [B*H*W]
        z_q = self.embedding(idx).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Straight-through
        z_q_st = z + (z_q - z).detach()

        if self.ema:
            with torch.no_grad():
                enc = F.one_hot(idx, num_classes=self.n_embed).type(flat.dtype)  # [B*H*W, V]
                self._ema_update(flat, enc)
            # EMA: commitment only
            vq_loss = self.beta * F.mse_loss(z.detach(), z_q)
        else:
            # Non-EMA: codebook + commitment
            codebook = F.mse_loss(z_q.detach(), z)
            commit   = F.mse_loss(z_q, z.detach())
            vq_loss = codebook + self.beta * commit

        return z_q_st, vq_loss, idx.view(B, H * W)


# =========================
# VQ-VAE wrapper
# =========================

class VQVAE(nn.Module):
    def __init__(self, vocab_size=512, z_channels=128, ch=128, beta=0.25, ema=True, ema_decay=0.99):
        super().__init__()
        self.encoder = Encoder(3, z_channels, ch)
        self.quantize = VectorQuantizer(vocab_size, z_channels, beta, ema, ema_decay)
        self.decoder = Decoder(3, z_channels, ch)
        self.vocab_size = vocab_size
        self.z_channels = z_channels

    def forward(self, x):
        z_e = self.encoder(x)  # [B, C, 8, 8]

        # latent normalization (safe scale) + clamp
        with torch.no_grad():
            std = z_e.detach().std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-3)
        z_e = z_e / std
        z_e = z_e.clamp(-4.0, 4.0)

        z_q, vq_loss, idx = self.quantize(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, idx

    @torch.no_grad()
    def img_to_idx(self, x):
        z_e = self.encoder(x)
        std = z_e.detach().std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-3)
        z_e = (z_e / std).clamp(-4.0, 4.0)
        _, _, idx = self.quantize(z_e)
        return idx  # [B, 64]

    @torch.no_grad()
    def idx_to_img(self, idx):
        """
        Convert a batch of token indices [B, 64] back to images [-1,1].
        Uses reshape (not view) to avoid contiguity issues.
        """
        # idx: [B, 64] (may be non-contiguous, may be non-long)
        idx = idx.to(torch.long).reshape(-1)  # [B*64]
        B = int(idx.numel() // 64)
        H = W = 8

        # Quantized latents
        z_q = self.quantize.embedding(idx)  # [B*64, C]
        z_q = z_q.reshape(B, H, W, self.z_channels).permute(0, 3, 1, 2).contiguous()

        # Decode to image in [-1, 1]
        x_hat = self.decoder(z_q)
        return x_hat


# =========================
# Training / Eval helpers
# =========================

def get_args():
    ap = argparse.ArgumentParser("Stable VQ-VAE on CIFAR-10")
    # model
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--z_channels", type=int, default=128)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.99)

    # train
    ap.add_argument("--data_path", type=str, default="./data")
    ap.add_argument("--output_dir", type=str, default="./output/vqvae_cifar10_stable")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lambda_mse", type=float, default=0.5)

    # amp / schedule
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--amp_warmup_epochs", type=int, default=5)
    ap.add_argument("--warmup_epochs", type=int, default=5)

    # logging
    ap.add_argument("--save_freq", type=int, default=10)
    ap.add_argument("--print_freq", type=int, default=100)
    return ap.parse_args()


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        prog = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    px = 3 * 32 * 32
    mse_sum = 0.0
    l1_sum = 0.0
    for x, _ in loader:
        x = x.to(device)
        x_hat, _, _ = model(x)
        mse_sum += F.mse_loss(x_hat, x, reduction="sum").item()
        l1_sum += F.l1_loss(x_hat, x, reduction="sum").item()
    mse = mse_sum / (len(loader.dataset) * px)
    l1 = l1_sum / (len(loader.dataset) * px)
    psnr = float(psnr_from_mse(torch.tensor(mse)))
    return float(mse), float(l1), psnr


@torch.no_grad()
def codebook_stats(model, loader, device, max_batches=10):
    model.eval()
    all_idx = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        idx = model.img_to_idx(x)  # [B, 64]
        all_idx.append(idx.flatten().cpu())
    all_idx = torch.cat(all_idx, dim=0)
    V = model.vocab_size
    unique = torch.unique(all_idx)
    usage = len(unique) / V
    counts = torch.bincount(all_idx, minlength=V).float()
    probs = counts / counts.sum().clamp_min(1.0)
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    perplexity = torch.exp(entropy)
    return float(usage), float(perplexity)


def save_recons(model, loader, device, outdir, tag="epoch"):
    model.eval()
    os.makedirs(outdir, exist_ok=True)
    x, _ = next(iter(loader))
    x = x[:32].to(device)
    with torch.no_grad():
        x_hat, _, _ = model(x)
    denorm = lambda t: t.add(1).mul(0.5).clamp(0, 1)
    grid = torch.cat([denorm(x), denorm(x_hat)], dim=0)
    save_image(grid, os.path.join(outdir, f"recon_{tag}.png"), nrow=8)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print("Stable VQ-VAE training on CIFAR-10")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    # Data
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=tf_train)
    test_set = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=tf_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Model
    model = VQVAE(vocab_size=args.vocab_size, z_channels=args.z_channels, ch=args.ch,
                  beta=args.beta, ema=args.ema, ema_decay=args.ema_decay).to(device)

    # Optim & sched
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    sched = cosine_with_warmup(optim, warmup_steps, total_steps)

    # AMP setup
    use_scaler = args.amp and (args.amp_dtype == "fp16")
    from torch import amp as torch_amp
    scaler = torch_amp.GradScaler('cuda', enabled=use_scaler)

    global_step = 0
    best_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"rec": 0.0, "vq": 0.0, "total": 0.0}

        # AMP warmup
        use_amp = args.amp and (epoch > args.amp_warmup_epochs)
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)

        for it, (x, _) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)

            with autocast_ctx:
                x_hat, vq_loss, _ = model(x)
                l1 = F.l1_loss(x_hat, x)
                mse = F.mse_loss(x_hat, x)
                rec_loss = l1 + args.lambda_mse * mse
                loss = rec_loss + vq_loss

            # Non-finite guard
            if not torch.isfinite(loss):
                print(f"⚠️ Non-finite loss at epoch {epoch}, step {global_step}. "
                      f"Stats: l1={float(l1)}, mse={float(mse)}, vq={float(vq_loss)}. "
                      f"Reducing LR x0.5 and disabling AMP for safety.")
                for g in optim.param_groups:
                    g["lr"] = g["lr"] * 0.5
                args.amp = False  # disable AMP from now
                use_amp = False
                # skip this step to avoid propagating NaNs
                continue

            optim.zero_grad(set_to_none=True)
            if use_amp and use_scaler:  # fp16 + scaler
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            sched.step()

            running["rec"] += float(rec_loss.item())
            running["vq"] += float(vq_loss.item())
            running["total"] += float(loss.item())

            if (global_step % args.print_freq) == 0:
                print(f"[Ep {epoch}/{args.epochs}] step {global_step} | "
                      f"rec={rec_loss.item():.4f} vq={vq_loss.item():.4f} total={loss.item():.4f} "
                      f"(AMP={'on' if use_amp else 'off'}:{args.amp_dtype})")
            global_step += 1

        nbt = len(train_loader)
        print(f"✓ Epoch {epoch} avg: rec={running['rec']/nbt:.4f} vq={running['vq']/nbt:.4f} total={running['total']/nbt:.4f}")

        # Eval
        mse_mean, l1_mean, psnr = eval_epoch(model, test_loader, device)
        usage, ppl = codebook_stats(model, test_loader, device)
        print(f"   Eval: MSE={mse_mean:.6f} | L1={l1_mean:.6f} | PSNR={psnr:.2f} dB | usage={usage*100:.2f}% | perplexity={ppl:.1f}")

        # Recon dump
        save_recons(model, test_loader, device, os.path.join(args.output_dir, "recon"), tag=f"ep{epoch:03d}")

        # Save
        if (epoch % args.save_freq == 0) or (epoch == args.epochs):
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "global_step": global_step
            }
            path = os.path.join(args.output_dir, f"vqvae_epoch{epoch:03d}.pth")
            torch.save(ckpt, path)
            print(f"   Saved: {path}")

        if mse_mean < best_mse:
            best_mse = mse_mean
            best_path = os.path.join(args.output_dir, "checkpoint_best.pth")
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, best_path)
            print(f"   ✓ New best (MSE={best_mse:.6f}) → {best_path}")

    print("✅ Training complete.")


if __name__ == "__main__":
    main()
