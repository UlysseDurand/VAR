"""
Training script for VQWGAN on CIFAR-10
Adapted for 32x32 images
"""
import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets

from models.vqwgan import VQWGAN
from utils.misc import MetricLogger
import dist


def get_args_parser():
    parser = argparse.ArgumentParser('VQWGAN CIFAR-10 training', add_help=False)
    
    # Model parameters (adapted for 32x32 images)
    parser.add_argument('--vocab_size', default=512, type=int, help='Smaller vocab for CIFAR-10')
    parser.add_argument('--z_channels', default=16, type=int, help='Reduced latent channels')
    parser.add_argument('--ch', default=64, type=int, help='Smaller base channels')
    parser.add_argument('--beta', default=0.25, type=float, help='commitment loss weight')
    parser.add_argument('--using_znorm', action='store_true')
    parser.add_argument('--quant_resi', default=0.5, type=float)
    parser.add_argument('--share_quant_resi', default=2, type=int, help='Fewer scales for 32x32')
    
    # WGAN parameters
    parser.add_argument('--disc_ch', default=64, type=int)
    parser.add_argument('--disc_num_layers', default=2, type=int, help='Fewer layers for small images')
    parser.add_argument('--disc_start', default=5000, type=int, help='Start disc earlier')
    parser.add_argument('--disc_weight', default=0.5, type=float)
    parser.add_argument('--gp_weight', default=10.0, type=float)
    parser.add_argument('--n_critic', default=5, type=int)
    parser.add_argument('--use_patch_disc', action='store_true')
    
    # Training parameters
    parser.add_argument('--batch_size', default=128, type=int, help='Larger batch for CIFAR')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Higher LR for small images')
    parser.add_argument('--disc_lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    
    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--workers', default=4, type=int)
    
    # Distributed training
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=1, type=int, help='GPU id to use (0 or 1)')
    
    # Logging
    parser.add_argument('--output_dir', default='./output/vqwgan_cifar10', type=str)
    parser.add_argument('--log_freq', default=50, type=int)
    parser.add_argument('--save_freq', default=2000, type=int)
    
    return parser


class VQWGANTrainer:
    def __init__(
        self,
        model: VQWGAN,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        device: torch.device,
        args,
    ):
        self.model = model
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.device = device
        self.args = args
        self.global_step = 0
    
    def train_step(
        self,
        real_imgs: torch.Tensor,
        metric_lg: MetricLogger,
    ) -> Tuple[float, float]:
        """Single training step"""
        real_imgs = real_imgs.to(self.device)
        
        # ========== Update Discriminator ==========
        if self.global_step >= self.args.disc_start:
            for _ in range(self.args.n_critic):
                self.optimizer_d.zero_grad()
                
                # Generate fake images
                with torch.no_grad():
                    fake_imgs, _, _ = self.model(real_imgs)
                
                # Compute discriminator loss
                d_loss, d_stats = self.model.discriminator_loss(real_imgs, fake_imgs)
                
                d_loss.backward()
                self.optimizer_d.step()
        else:
            d_loss = torch.tensor(0.0)
            d_stats = {}
        
        # ========== Update Generator ==========
        self.optimizer_g.zero_grad()
        
        # Compute generator loss
        g_total_loss, g_stats = self.model.compute_loss(real_imgs, self.global_step)
        
        g_total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer_g.step()
        
        # Update metrics
        if self.global_step % self.args.log_freq == 0:
            metric_lg.update(
                g_loss=g_stats['total_loss'],
                rec_loss=g_stats['rec_loss'],
                vq_loss=g_stats['vq_loss'],
                adv_loss=g_stats['g_loss'],
            )
            
            if d_stats:
                metric_lg.update(
                    d_loss=d_loss.item(),
                    d_real=d_stats.get('disc_real', 0),
                    d_fake=d_stats.get('disc_fake', 0),
                    gp=d_stats.get('gradient_penalty', 0),
                )
        
        self.global_step += 1
        
        return g_total_loss.item(), d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss
    
    @torch.no_grad()
    def visualize(self, val_loader: DataLoader, save_path: str):
        """Visualize reconstructions"""
        self.model.eval()
        
        real_imgs = next(iter(val_loader))[0][:8].to(self.device)
        rec_imgs, _, _ = self.model(real_imgs)
        
        # Concatenate real and reconstructed
        comparison = torch.cat([real_imgs, rec_imgs], dim=0)
        
        # Save grid
        grid = torchvision.utils.make_grid(comparison, nrow=8, normalize=True, value_range=(-1, 1))
        torchvision.utils.save_image(grid, save_path)
        
        self.model.train()


def main(args):
    # Select GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"✓ Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    
    # Setup distributed mode (required even for single GPU)
    if not torch.distributed.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        torch.distributed.init_process_group(
            backend='gloo' if not torch.cuda.is_available() else 'nccl',
            init_method='env://',
            world_size=1,
            rank=0
        )
        print("✓ Initialized single-process distributed mode")
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # CIFAR-10 specific: smaller patch numbers for 32x32 images
    # For 32x32: downsample 16x → 2x2 final size
    v_patch_nums = (1, 2)  # Only 2 scales for 32x32
    
    # Build model
    print(f'Building VQWGAN model for CIFAR-10 (32x32)...')
    model = VQWGAN(
        vocab_size=args.vocab_size,
        z_channels=args.z_channels,
        ch=args.ch,
        beta=args.beta,
        using_znorm=args.using_znorm,
        quant_resi=args.quant_resi,
        share_quant_resi=args.share_quant_resi,
        v_patch_nums=v_patch_nums,
        disc_ch=args.disc_ch,
        disc_num_layers=args.disc_num_layers,
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        gp_weight=args.gp_weight,
        use_patch_disc=args.use_patch_disc,
        test_mode=False,
    )
    model = model.to(device)
    
    # Build optimizers
    optimizer_g = torch.optim.AdamW(
        [
            {'params': model.encoder.parameters()},
            {'params': model.decoder.parameters()},
            {'params': model.quantize.parameters()},
            {'params': model.quant_conv.parameters()},
            {'params': model.post_quant_conv.parameters()},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    disc_params = [{'params': model.discriminator.parameters()}]
    if model.patch_discriminator is not None:
        disc_params.append({'params': model.patch_discriminator.parameters()})
    
    optimizer_d = torch.optim.Adam(
        disc_params,
        lr=args.disc_lr,
        betas=(0.5, 0.9),
    )
    
    # Build CIFAR-10 dataset
    print(f'Loading CIFAR-10 from {args.data_path}...')
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=args.data_path,
        train=False,
        download=True,
        transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    # Build trainer
    trainer = VQWGANTrainer(
        model=model,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=device,
        args=args,
    )
    
    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print(f'Starting training for {args.epochs} epochs...')
    print(f'Dataset: CIFAR-10 (50k train, 10k test)')
    print(f'Image size: 32x32')
    print(f'Batch size: {args.batch_size}')
    print(f'Vocab size: {args.vocab_size}')
    print(f'Discriminator starts at step: {args.disc_start}')
    
    for epoch in range(args.epochs):
        model.train()
        metric_lg = MetricLogger(delimiter='  ')  # Create new logger per epoch
        
        for it, (imgs, _) in enumerate(train_loader):
            g_loss, d_loss = trainer.train_step(imgs, metric_lg)
            
            # Logging
            if trainer.global_step % args.log_freq == 0:
                print(f'Epoch [{epoch}/{args.epochs}] '
                      f'Step [{trainer.global_step}] '
                      f'G_loss: {g_loss:.4f} D_loss: {d_loss:.4f}')
            
            # Save checkpoint
            if trainer.global_step % args.save_freq == 0 and trainer.global_step > 0:
                save_path = os.path.join(args.output_dir, f'vqwgan_step{trainer.global_step}.pth')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(),
                    'optimizer_d': optimizer_d.state_dict(),
                    'epoch': epoch,
                    'global_step': trainer.global_step,
                    'args': args,
                }, save_path)
                print(f'✓ Saved checkpoint to {save_path}')
                
                # Visualize
                vis_path = os.path.join(args.output_dir, f'recon_step{trainer.global_step}.png')
                trainer.visualize(test_loader, vis_path)
                print(f'✓ Saved visualization to {vis_path}')
        
        # End of epoch
        save_path = os.path.join(args.output_dir, f'vqwgan_epoch{epoch}.pth')
        torch.save({
            'model': model.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'epoch': epoch,
            'global_step': trainer.global_step,
            'args': args,
        }, save_path)
        print(f'✓ Saved epoch checkpoint to {save_path}')
    
    print('Training completed!')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
