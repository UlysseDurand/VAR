"""
Utility script for VQWGAN
- Convert VQVAE checkpoint to VQWGAN
- Evaluate reconstruction quality
- Compare VQVAE vs VQWGAN
"""
import argparse
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from models import VQVAE
from models.vqwgan import VQWGAN


def convert_vqvae_to_vqwgan(vqvae_path, output_path, config):
    """
    Convert pretrained VQVAE checkpoint to VQWGAN format
    """
    print(f"Loading VQVAE from {vqvae_path}")
    
    # Load VQVAE
    vqvae = VQVAE(**config, test_mode=True)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location='cpu'))
    
    # Create VQWGAN with same config
    print("Creating VQWGAN with same architecture")
    vqwgan = VQWGAN(
        **config,
        disc_ch=64,
        disc_num_layers=3,
        disc_start=10000,
        disc_weight=0.5,
        gp_weight=10.0,
        use_patch_disc=True,
        test_mode=False,
    )
    
    # Transfer weights
    print("Transferring weights from VQVAE to VQWGAN")
    vqwgan.encoder.load_state_dict(vqvae.encoder.state_dict())
    vqwgan.decoder.load_state_dict(vqvae.decoder.state_dict())
    vqwgan.quantize.load_state_dict(vqvae.quantize.state_dict())
    vqwgan.quant_conv.load_state_dict(vqvae.quant_conv.state_dict())
    vqwgan.post_quant_conv.load_state_dict(vqvae.post_quant_conv.state_dict())
    
    # Save VQWGAN checkpoint
    print(f"Saving VQWGAN to {output_path}")
    torch.save({
        'model': vqwgan.state_dict(),
        'config': config,
    }, output_path)
    
    print("✓ Conversion completed successfully!")
    return vqwgan


def evaluate_reconstruction(model, data_loader, device, num_samples=1000):
    """
    Evaluate reconstruction quality on a dataset
    """
    model.eval()
    model.to(device)
    
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    print(f"Evaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(data_loader)):
            if total_samples >= num_samples:
                break
            
            images = images.to(device)
            
            # Reconstruct
            if isinstance(model, VQWGAN):
                reconstructed, _, vq_loss = model(images, ret_usages=False)
            else:
                reconstructed, _, vq_loss = model(images, ret_usages=False)
            
            # Compute metrics
            mse = F.mse_loss(reconstructed, images, reduction='sum')
            mae = F.l1_loss(reconstructed, images, reduction='sum')
            
            batch_size = images.size(0)
            total_mse += mse.item()
            total_mae += mae.item()
            total_samples += batch_size
    
    # Average metrics
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    # PSNR (assuming images in [-1, 1])
    max_val = 2.0
    psnr = 10 * torch.log10(torch.tensor(max_val**2 / avg_mse)).item()
    
    results = {
        'MSE': avg_mse,
        'MAE': avg_mae,
        'PSNR': psnr,
        'num_samples': total_samples,
    }
    
    return results


def compare_models(vqvae, vqwgan, data_loader, device, num_samples=1000):
    """
    Compare VQVAE and VQWGAN reconstruction quality
    """
    print("=" * 70)
    print("COMPARING VQVAE vs VQWGAN")
    print("=" * 70)
    
    # Evaluate VQVAE
    print("\nEvaluating VQVAE...")
    vqvae_results = evaluate_reconstruction(vqvae, data_loader, device, num_samples)
    
    print("\nVQVAE Results:")
    print(f"  MSE:  {vqvae_results['MSE']:.6f}")
    print(f"  MAE:  {vqvae_results['MAE']:.6f}")
    print(f"  PSNR: {vqvae_results['PSNR']:.2f} dB")
    
    # Evaluate VQWGAN
    print("\nEvaluating VQWGAN...")
    vqwgan_results = evaluate_reconstruction(vqwgan, data_loader, device, num_samples)
    
    print("\nVQWGAN Results:")
    print(f"  MSE:  {vqwgan_results['MSE']:.6f}")
    print(f"  MAE:  {vqwgan_results['MAE']:.6f}")
    print(f"  PSNR: {vqwgan_results['PSNR']:.2f} dB")
    
    # Compute improvements
    print("\n" + "=" * 70)
    print("IMPROVEMENTS")
    print("=" * 70)
    
    mse_improvement = ((vqvae_results['MSE'] - vqwgan_results['MSE']) / vqvae_results['MSE']) * 100
    mae_improvement = ((vqvae_results['MAE'] - vqwgan_results['MAE']) / vqvae_results['MAE']) * 100
    psnr_improvement = vqwgan_results['PSNR'] - vqvae_results['PSNR']
    
    print(f"MSE:  {mse_improvement:+.2f}%")
    print(f"MAE:  {mae_improvement:+.2f}%")
    print(f"PSNR: {psnr_improvement:+.2f} dB")
    
    return {
        'vqvae': vqvae_results,
        'vqwgan': vqwgan_results,
        'improvements': {
            'mse_pct': mse_improvement,
            'mae_pct': mae_improvement,
            'psnr_db': psnr_improvement,
        }
    }


def analyze_codebook_usage(model, data_loader, device, num_samples=10000):
    """
    Analyze codebook usage statistics
    """
    model.eval()
    model.to(device)
    
    vocab_size = model.vocab_size
    codebook_hits = torch.zeros(vocab_size, dtype=torch.long)
    
    print(f"Analyzing codebook usage on {num_samples} samples...")
    
    total_samples = 0
    with torch.no_grad():
        for images, _ in tqdm(data_loader):
            if total_samples >= num_samples:
                break
            
            images = images.to(device)
            
            # Get token indices
            idx_Bl = model.img_to_idxBl(images)
            
            # Count token usage (only last scale for simplicity)
            idx_last = idx_Bl[-1]  # [B, L]
            for idx in idx_last.view(-1).cpu():
                codebook_hits[idx] += 1
            
            total_samples += images.size(0)
    
    # Compute statistics
    total_tokens = codebook_hits.sum().item()
    used_tokens = (codebook_hits > 0).sum().item()
    usage_rate = (used_tokens / vocab_size) * 100
    
    # Compute entropy (diversity measure)
    probs = codebook_hits.float() / total_tokens
    probs = probs[probs > 0]  # Remove zeros
    entropy = -(probs * torch.log(probs)).sum().item()
    max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float)).item()
    normalized_entropy = entropy / max_entropy
    
    results = {
        'vocab_size': vocab_size,
        'used_tokens': used_tokens,
        'usage_rate': usage_rate,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'total_tokens': total_tokens,
        'num_samples': total_samples,
    }
    
    print("\n" + "=" * 70)
    print("CODEBOOK USAGE STATISTICS")
    print("=" * 70)
    print(f"Vocabulary size:      {vocab_size}")
    print(f"Used tokens:          {used_tokens} ({usage_rate:.2f}%)")
    print(f"Entropy:              {entropy:.4f} (max: {max_entropy:.4f})")
    print(f"Normalized entropy:   {normalized_entropy:.4f}")
    print(f"Total tokens sampled: {total_tokens}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='VQWGAN utilities')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['convert', 'evaluate', 'compare', 'codebook'],
                      help='Operation mode')
    
    # Model paths
    parser.add_argument('--vqvae_path', type=str, default='vae_ch160v4096z32.pth',
                      help='Path to VQVAE checkpoint')
    parser.add_argument('--vqwgan_path', type=str, default='vqwgan.pth',
                      help='Path to VQWGAN checkpoint')
    parser.add_argument('--output_path', type=str, default='vqwgan_converted.pth',
                      help='Output path for converted model')
    
    # Model config
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--z_channels', type=int, default=32)
    parser.add_argument('--ch', type=int, default=160)
    
    # Evaluation
    parser.add_argument('--data_path', type=str, default='./data/imagenet')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Model config
    config = {
        'vocab_size': args.vocab_size,
        'z_channels': args.z_channels,
        'ch': args.ch,
        'dropout': 0.0,
        'beta': 0.25,
        'using_znorm': False,
        'quant_conv_ks': 3,
        'quant_resi': 0.5,
        'share_quant_resi': 4,
        'v_patch_nums': (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    }
    
    if args.mode == 'convert':
        # Convert VQVAE to VQWGAN
        vqwgan = convert_vqvae_to_vqwgan(args.vqvae_path, args.output_path, config)
        
    elif args.mode in ['evaluate', 'compare', 'codebook']:
        # Load data
        from utils.data import build_dataset
        from torch.utils.data import DataLoader
        
        print(f"Loading dataset from {args.data_path}")
        dataset = build_dataset(args.data_path, final_reso=256, is_train=False)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        
        if args.mode == 'evaluate':
            # Evaluate single model
            model_path = args.vqwgan_path if os.path.exists(args.vqwgan_path) else args.vqvae_path
            model_type = 'VQWGAN' if 'vqwgan' in model_path else 'VQVAE'
            
            print(f"Loading {model_type} from {model_path}")
            if model_type == 'VQWGAN':
                model = VQWGAN(**config, test_mode=True)
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
            else:
                model = VQVAE(**config, test_mode=True)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            results = evaluate_reconstruction(model, data_loader, args.device, args.num_samples)
            
            print("\n" + "=" * 70)
            print(f"{model_type} EVALUATION RESULTS")
            print("=" * 70)
            print(f"MSE:  {results['MSE']:.6f}")
            print(f"MAE:  {results['MAE']:.6f}")
            print(f"PSNR: {results['PSNR']:.2f} dB")
            
        elif args.mode == 'compare':
            # Compare VQVAE and VQWGAN
            print("Loading VQVAE...")
            vqvae = VQVAE(**config, test_mode=True)
            vqvae.load_state_dict(torch.load(args.vqvae_path, map_location='cpu'))
            
            print("Loading VQWGAN...")
            vqwgan = VQWGAN(**config, test_mode=True)
            checkpoint = torch.load(args.vqwgan_path, map_location='cpu')
            vqwgan.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
            
            results = compare_models(vqvae, vqwgan, data_loader, args.device, args.num_samples)
            
        elif args.mode == 'codebook':
            # Analyze codebook usage
            model_path = args.vqwgan_path if os.path.exists(args.vqwgan_path) else args.vqvae_path
            model_type = 'VQWGAN' if 'vqwgan' in model_path else 'VQVAE'
            
            print(f"Loading {model_type} from {model_path}")
            if model_type == 'VQWGAN':
                model = VQWGAN(**config, test_mode=True)
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
            else:
                model = VQVAE(**config, test_mode=True)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            results = analyze_codebook_usage(model, data_loader, args.device, args.num_samples)


if __name__ == '__main__':
    main()
