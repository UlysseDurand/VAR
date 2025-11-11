
import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models.vqwgan import VQWGAN
from models.var import VAR



def get_args():
    parser = argparse.ArgumentParser('VAR training on CIFAR-10 with VQWGAN')

    # Paths
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--vqwgan_path', type=str, required=True, help='Path to trained VQWGAN checkpoint')
    parser.add_argument('--output_dir', type=str, default='./output/var_vqwgan_cifar10')

    # VAR model
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--attn_drop_rate', type=float, default=0.1)

    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--label_smooth', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Logging / Eval
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=5)

    # Sampling
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--cfg_scale', type=float, default=1.5)
    parser.add_argument('--top_k', type=int, default=512)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)

    return parser.parse_args()


# Loss cross entropy 
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        # pred: [B, V], target: [B]
        n_classes = pred.size(1)
        log_pred = F.log_softmax(pred, dim=-1)
        target_one_hot = torch.zeros_like(log_pred).scatter_(1, target.unsqueeze(1), 1)
        target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        loss = -(target_smooth * log_pred).sum(dim=-1).mean()
        return loss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps): # tip to decease LR seen on VAR repo
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


#util to convert v_patch_nums to tuple of int
def _to_tuple_int(v):
    if isinstance(v, str):
        return tuple(int(x.strip()) for x in v.split(',') if x.strip())
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    return (1, 2, 4, 8)


#load VQWGAN
def load_vqwgan(args, device):
    print(f"📥 Loading VQWGAN from {args.vqwgan_path}...")
    if not os.path.exists(args.vqwgan_path):
        raise FileNotFoundError(args.vqwgan_path)

    try:
        checkpoint = torch.load(args.vqwgan_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.vqwgan_path, map_location=device)

    # defaults
    vocab_size = 1024
    z_channels = 32
    ch = 128
    beta = 0.25
    quant_resi = 0.5
    share_quant_resi = 4
    v_patch_nums = (1, 2, 4, 8)

    if 'args' in checkpoint:
        vqa = checkpoint['args']
        src = vars(vqa) if hasattr(vqa, '__dict__') else (vqa if isinstance(vqa, dict) else {})
        vocab_size = src.get('vocab_size', vocab_size)
        z_channels = src.get('z_channels', z_channels)
        ch = src.get('ch', ch)
        beta = src.get('beta', beta)
        quant_resi = src.get('quant_resi', quant_resi)
        share_quant_resi = src.get('share_quant_resi', share_quant_resi)
        v_patch_nums = _to_tuple_int(src.get('v_patch_nums', v_patch_nums))
        print("Using VQWGAN config from checkpoint")
    else:
        print("No config found in VQWGAN checkpoint; using CIFAR-10 defaults")

    print(f"   vocab_size={vocab_size}, z_channels={z_channels}, ch={ch}, beta={beta}, "
          f"quant_resi={quant_resi}, share_quant_resi={share_quant_resi}, v_patch_nums={v_patch_nums}")

    vqwgan = VQWGAN(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=ch,
        beta=beta,
        quant_resi=quant_resi,
        share_quant_resi=share_quant_resi,
        v_patch_nums=v_patch_nums,
        test_mode=True,
    ).to(device)

    # choose state dict
    if 'generator' in checkpoint:
        state = checkpoint['generator']
        print("Loading VQ params from checkpoint['generator']")
    elif 'vqwgan' in checkpoint:
        state = checkpoint['vqwgan']
        print("Loading VQ params from checkpoint['vqwgan']")
    elif 'model' in checkpoint:
        state = checkpoint['model']
        print("Loading VQ params from checkpoint['model']")
    else:
        state = checkpoint
        print("Loading VQ params from checkpoint root")

    state = {k: v for k, v in state.items()
             if not (k.startswith('discriminator.') or k.startswith('patch_discriminator.'))}
    incompat = vqwgan.load_state_dict(state, strict=False)
    if getattr(incompat, 'missing_keys', None):
        print(f"Missing VQ keys: {len(incompat.missing_keys)} (showing first 10)")
        for k in incompat.missing_keys[:10]:
            print(f"   - {k}")
    if getattr(incompat, 'unexpected_keys', None):
        non_disc = [k for k in incompat.unexpected_keys if 'discriminator' not in k]
        if non_disc:
            print(f"Unexpected non-D VQ keys: {len(non_disc)}")
            for k in non_disc[:10]:
                print(f"   - {k}")

    vqwgan.eval()
    for p in vqwgan.parameters():
        p.requires_grad = False
    return vqwgan


# =========================
# VAR
# =========================
def create_var_model(vqwgan, args, device):
    print("Creating VAR..")
    patch_nums = getattr(vqwgan.quantize, 'v_patch_nums', (1, 2, 4, 8)) # 4 scales for cifar10
    var = VAR(
        vae_local=vqwgan,
        num_classes=10,
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=0.1,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        patch_nums=patch_nums,
        flash_if_available=True,
        fused_if_available=True,
    ).to(device)
    print("VAR ready")
    return var


#data loader
def prepare_data(args):
    print("Preparing CIFAR-10 dataset...")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("Dataset ready")
    return train_loader, test_loader


def _concat_targets(gt_ms_idx_Bl):
    return torch.cat(gt_ms_idx_Bl, dim=1)  # [B, L_total]

def _scale_lengths(gt_ms_idx_Bl):
    return [t.size(1) for t in gt_ms_idx_Bl]  # [L1, L2, ...]


def _split_by_scales(tensor_BL, lengths):
    # tensor_BL: [B, L_total] -> list of [B, Li]
    out = []
    start = 0
    for L in lengths:
        out.append(tensor_BL[:, start:start+L])
        start += L
    return out


# Train / Eval 
def train_one_epoch(var, vqwgan, train_loader, optimizer, scheduler, criterion, epoch, args, device):
    var.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(train_loader)

    # per-scale stats
    sum_acc_scales = None  # list of sums
    sum_cnt_scales = None

    t0 = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            gt_ms_idx_Bl = vqwgan.img_to_idxBl(images)  # list of [B, Li]
            lengths = _scale_lengths(gt_ms_idx_Bl)
            target_BL = _concat_targets(gt_ms_idx_Bl)   # [B, L]

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        logits_BLV = var(labels, vqwgan.quantize.idxBl_to_var_input(gt_ms_idx_Bl))  # [B, L, V]
        loss = criterion(logits_BLV.reshape(-1, logits_BLV.size(-1)), target_BL.reshape(-1))

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(var.parameters(), args.grad_clip)
        optimizer.step()

        scheduler.step()

        with torch.no_grad():
            pred_BL = logits_BLV.argmax(dim=-1)  # [B, L]
            acc = (pred_BL == target_BL).float().mean()

            # per-scale accuracy
            pred_list = _split_by_scales(pred_BL, lengths)
            tgt_list = gt_ms_idx_Bl
            scale_accs = [(p == t).float().mean().item() for p, t in zip(pred_list, tgt_list)]
            if sum_acc_scales is None:
                sum_acc_scales = [0.0] * len(scale_accs)
                sum_cnt_scales = [0] * len(scale_accs)
            for i, a in enumerate(scale_accs):
                sum_acc_scales[i] += a
                sum_cnt_scales[i] += 1

        total_loss += loss.item()
        total_acc += acc.item()

        if batch_idx % args.print_freq == 0:
            dt = time.time() - t0
            lr = optimizer.param_groups[0]['lr']
            scale_str = " ".join([f"s{i}:{sum_acc_scales[i]/max(1,sum_cnt_scales[i]):.3f}" for i in range(len(sum_acc_scales))])
            print(f"Epoch[{epoch}] {batch_idx:04d}/{num_batches} "
                  f"Loss:{loss.item():.4f} Acc:{acc.item():.4f} LR:{lr:.6f} {scale_str}  ({dt:.2f}s)")
            t0 = time.time()

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_scales = [sum_acc_scales[i] / max(1, sum_cnt_scales[i]) for i in range(len(sum_acc_scales))]
    return avg_loss, avg_acc, avg_scales


@torch.no_grad()
def evaluate(var, vqwgan, test_loader, criterion, args, device):
    var.eval()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(test_loader)

    sum_acc_scales = None
    sum_cnt_scales = None

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        gt_ms_idx_Bl = vqwgan.img_to_idxBl(images)
        lengths = _scale_lengths(gt_ms_idx_Bl)
        target_BL = _concat_targets(gt_ms_idx_Bl)

        logits_BLV = var(labels, vqwgan.quantize.idxBl_to_var_input(gt_ms_idx_Bl))
        loss = criterion(logits_BLV.reshape(-1, logits_BLV.size(-1)), target_BL.reshape(-1))

        pred_BL = logits_BLV.argmax(dim=-1)
        acc = (pred_BL == target_BL).float().mean()

        # per-scale accuracy
        pred_list = _split_by_scales(pred_BL, lengths)
        scale_accs = [(p == t).float().mean().item() for p, t in zip(pred_list, gt_ms_idx_Bl)]
        if sum_acc_scales is None:
            sum_acc_scales = [0.0] * len(scale_accs)
            sum_cnt_scales = [0] * len(scale_accs)
        for i, a in enumerate(scale_accs):
            sum_acc_scales[i] += a
            sum_cnt_scales[i] += 1

        total_loss += loss.item()
        total_acc += acc.item()

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_scales = [sum_acc_scales[i] / max(1, sum_cnt_scales[i]) for i in range(len(sum_acc_scales))]
    return avg_loss, avg_acc, avg_scales


#sampling funcs
@torch.no_grad()
def sample_images(var, vqwgan, epoch, args, device):
    var.eval()

    print(f"Generating {args.num_samples} samples...")
    num_per_class = max(1, args.num_samples // 10)
    all_samples = []
    vocab = getattr(vqwgan, 'vocab_size', 1024)
    top_k = max(1, min(args.top_k, vocab - 1))

    for class_id in range(10):
        labels = torch.full((num_per_class,), class_id, dtype=torch.long, device=device)
        # handle temperature support gracefully
        try:
            imgs = var.autoregressive_infer_cfg(
                B=num_per_class,
                label_B=labels,
                cfg=args.cfg_scale,
                top_k=top_k,
                top_p=args.top_p,
                g_seed=epoch * 10 + class_id,
                more_smooth=False,
                temperature=args.temperature,
            )
        except TypeError:
            imgs = var.autoregressive_infer_cfg(
                B=num_per_class,
                label_B=labels,
                cfg=args.cfg_scale,
                top_k=top_k,
                top_p=args.top_p,
                g_seed=epoch * 10 + class_id,
                more_smooth=False,
            )
        all_samples.append(imgs)

    all_samples = torch.cat(all_samples, dim=0).clamp(0, 1)
    save_path = os.path.join(args.output_dir, 'samples', f'epoch_{epoch:04d}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    nrow = 10
    if int(args.num_samples ** 0.5) ** 2 == args.num_samples:
        nrow = int(args.num_samples ** 0.5)
    save_image(all_samples, save_path, nrow=nrow)
    print(f"Samples saved to {save_path}")

    return all_samples


@torch.no_grad()
def save_reconstructions(vqwgan, test_loader, epoch, args, device):
    ds = test_loader.dataset
    n = len(ds)
    start = (epoch * 37) % (n - 32) if n > 32 else 0 # 37 arbitrary prime but  > 32
    imgs = torch.stack([ds[i][0] for i in range(start, start + 32)]).to(device)

    rec = vqwgan.img_to_reconstructed_img(imgs, last_one=True)

    imgs_vis = imgs.add(1).mul(0.5).clamp(0, 1)
    rec_vis = rec.add(1).mul(0.5).clamp(0, 1)
    comp = torch.cat([imgs_vis, rec_vis], dim=0)

    save_path = os.path.join(args.output_dir, 'reconstructions', f'epoch_{epoch:04d}_recon.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(comp, save_path, nrow=8)
    print(f"Reconstructions saved to {save_path}")


# Main

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("VAR Training with VQWGAN (CIFAR-10)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")

    vqwgan = load_vqwgan(args, device)
    var = create_var_model(vqwgan, args, device)

    train_loader, test_loader = prepare_data(args)

    optimizer = torch.optim.AdamW(
        var.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smooth)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 80}\nEpoch {epoch}/{args.epochs}\n{'=' * 80}")

        train_loss, train_acc, train_scales = train_one_epoch(
            var, vqwgan, train_loader, optimizer, scheduler, criterion, epoch, args, device
        )
        print(f"\nTrain | Loss:{train_loss:.4f}  Acc:{train_acc:.4f}  "
              f"Per-scale:{' '.join([f'{a:.3f}' for a in train_scales])}")

        if epoch % args.eval_freq == 0:
            val_loss, val_acc, val_scales = evaluate(var, vqwgan, test_loader, criterion, args, device)
            print(f"Val   | Loss:{val_loss:.4f}  Acc:{val_acc:.4f}  "
                  f"Per-scale:{' '.join([f'{a:.3f}' for a in val_scales])}")

            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model': var.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'args': vars(args),
                }, save_path)
                print(f"Best model saved (acc: {best_acc:.4f})")

            sample_images(var, vqwgan, epoch, args, device)
            save_reconstructions(vqwgan, test_loader, epoch, args, device)

        if epoch % args.save_freq == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch:04d}.pth')
            torch.save({
                'epoch': epoch,
                'model': var.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'args': vars(args),
            }, save_path)
            print(f"Checkpoint saved: {save_path}")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
