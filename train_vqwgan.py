"""
Training script for VQWGAN
Alternates between generator (VAE) and discriminator updates
"""
import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from models.vqwgan import VQWGAN
from utils.data import build_dataset
from utils.misc import MetricLogger, TensorboardLogger
import dist


def get_args_parser():
    parser = argparse.ArgumentParser('VQWGAN training', add_help=False)
    
    # Model parameters
    parser.add_argument('--vocab_size', default=4096, type=int)
    parser.add_argument('--z_channels', default=32, type=int)
    parser.add_argument('--ch', default=160, type=int)
    parser.add_argument('--beta', default=0.25, type=float, help='commitment loss weight')
    parser.add_argument('--using_znorm', action='store_true')
    parser.add_argument('--quant_resi', default=0.5, type=float)
    parser.add_argument('--share_quant_resi', default=4, type=int)
    
    # WGAN parameters
    parser.add_argument('--disc_ch', default=64, type=int, help='discriminator base channels')
    parser.add_argument('--disc_num_layers', default=3, type=int)
    parser.add_argument('--disc_start', default=10000, type=int, help='when to start disc training')
    parser.add_argument('--disc_weight', default=0.5, type=float, help='adversarial loss weight')
    parser.add_argument('--gp_weight', default=10.0, type=float, help='gradient penalty weight')
    parser.add_argument('--n_critic', default=5, type=int, help='critic updates per generator update')
    parser.add_argument('--use_patch_disc', action='store_true', help='use patch discriminator')
    
    # Training parameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=4.5e-6, type=float)
    parser.add_argument('--disc_lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    
    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--workers', default=8, type=int)
    
    # Distributed training
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--seed', default=0, type=int)
    
    # Logging
    parser.add_argument('--output_dir', default='./output/vqwgan', type=str)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=5000, type=int)
    
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
        """
        Single training step with alternating updates
        """
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
        
        # ========== Update Generator (VAE) ==========
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
        """
        Visualize reconstructions
        """
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
    # Setup distributed training
    dist.init_distributed_mode(args)
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Build model
    print(f'Building VQWGAN model...')
    model = VQWGAN(
        vocab_size=args.vocab_size,
        z_channels=args.z_channels,
        ch=args.ch,
        beta=args.beta,
        using_znorm=args.using_znorm,
        quant_resi=args.quant_resi,
        share_quant_resi=args.share_quant_resi,
        disc_ch=args.disc_ch,
        disc_num_layers=args.disc_num_layers,
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        gp_weight=args.gp_weight,
        use_patch_disc=args.use_patch_disc,
        test_mode=False,
    )
    model = model.to(device)
    
    if dist.is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=False
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Build optimizers
    # Generator (VAE) optimizer
    optimizer_g = torch.optim.AdamW(
        [
            {'params': model_without_ddp.encoder.parameters()},
            {'params': model_without_ddp.decoder.parameters()},
            {'params': model_without_ddp.quantize.parameters()},
            {'params': model_without_ddp.quant_conv.parameters()},
            {'params': model_without_ddp.post_quant_conv.parameters()},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    # Discriminator optimizer
    optimizer_d = torch.optim.Adam(
        [
            {'params': model_without_ddp.discriminator.parameters()},
        ] + (
            [{'params': model_without_ddp.patch_discriminator.parameters()}]
            if model_without_ddp.patch_discriminator is not None else []
        ),
        lr=args.disc_lr,
        betas=(0.5, 0.9),
    )
    
    # Build dataset
    print(f'Building dataset from {args.data_path}...')
    train_dataset = build_dataset(
        args.data_path,
        final_reso=args.img_size,
        is_train=True,
    )
    
    if dist.is_dist_avail_and_initialized():
        sampler = torch.utils.data.DistributedSampler(
            train_dataset, shuffle=True, seed=args.seed
        )
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Build trainer
    trainer = VQWGANTrainer(
        model=model_without_ddp,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=device,
        args=args,
    )
    
    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    metric_lg = MetricLogger(delimiter='  ')
    tb_lg = TensorboardLogger(args.output_dir) if dist.is_master() else None
    
    # Training loop
    print(f'Starting training for {args.epochs} epochs...')
    for epoch in range(args.epochs):
        if dist.is_dist_avail_and_initialized():
            sampler.set_epoch(epoch)
        
        model.train()
        metric_lg.reset()
        
        for it, (imgs, _) in enumerate(train_loader):
            g_loss, d_loss = trainer.train_step(imgs, metric_lg)
            
            # Logging
            if trainer.global_step % args.log_freq == 0 and dist.is_master():
                print(f'Epoch [{epoch}/{args.epochs}] Iter [{it}/{len(train_loader)}] '
                      f'G_loss: {g_loss:.4f} D_loss: {d_loss:.4f}')
                
                if tb_lg is not None:
                    tb_lg.update(head='train', step=trainer.global_step, **metric_lg.meters)
            
            # Save checkpoint
            if trainer.global_step % args.save_freq == 0 and dist.is_master():
                save_path = os.path.join(args.output_dir, f'vqwgan_step{trainer.global_step}.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(),
                    'optimizer_d': optimizer_d.state_dict(),
                    'epoch': epoch,
                    'global_step': trainer.global_step,
                    'args': args,
                }, save_path)
                print(f'Saved checkpoint to {save_path}')
                
                # Visualize
                vis_path = os.path.join(args.output_dir, f'recon_step{trainer.global_step}.png')
                trainer.visualize(train_loader, vis_path)
        
        # End of epoch
        if dist.is_master():
            save_path = os.path.join(args.output_dir, f'vqwgan_epoch{epoch}.pth')
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'epoch': epoch,
                'global_step': trainer.global_step,
                'args': args,
            }, save_path)
            print(f'Saved epoch checkpoint to {save_path}')
    
    print('Training completed!')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
