#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génération d'images CIFAR-10 par classe avec VAR + VQWGAN (déjà entraînés).
- Recharge la config VQWGAN depuis le checkpoint (beta, quant_resi, share_quant_resi, v_patch_nums).
- Construit le VAR, charge le checkpoint.
- Génère N images pour une classe demandée (0..9 ou nom de classe).
- Sauvegarde une grille + images individuelles.

Exemples :
  python gen_var_class.py \
    --vqwgan_ckpt ./output/vqwgan_ultra_stable/vqwgan_epoch29.pth \
    --var_ckpt ./output/var_vqwgan_cifar10/checkpoint_best.pth \
    --class car --num 64 --cfg 1.5 --top_k 512 --top_p 0.9 --seed 123 \
    --outdir ./output/var_samples

  python gen_var_class.py \
    --vqwgan_ckpt ... --var_ckpt ... --class 7 --num 40
"""

import os
import argparse
import torch
import torchvision
from torchvision.utils import save_image

from models.vqwgan import VQWGAN
from models.var import VAR

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]
NAME_TO_ID = {
    "airplane":0, "plane":0,
    "automobile":1, "car":1, "auto":1,
    "bird":2,
    "cat":3,
    "deer":4,
    "dog":5,
    "frog":6,
    "horse":7,
    "ship":8, "boat":8,
    "truck":9, "lorry":9,
}

def _to_tuple_int(v):
    if isinstance(v, str):
        return tuple(int(x.strip()) for x in v.split(',') if x.strip())
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    return (1,2,4,8)

def load_vqwgan(vqwgan_ckpt, device):
    if not os.path.exists(vqwgan_ckpt):
        raise FileNotFoundError(vqwgan_ckpt)
    try:
        ckpt = torch.load(vqwgan_ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(vqwgan_ckpt, map_location=device)

    # Defaults
    cfg = dict(
        vocab_size=1024, z_channels=32, ch=128,
        beta=0.25, quant_resi=0.5, share_quant_resi=4,
        v_patch_nums=(1,2,4,8)
    )
    if 'args' in ckpt:
        args = ckpt['args']
        src = vars(args) if hasattr(args, '__dict__') else (args if isinstance(args, dict) else {})
        cfg['vocab_size'] = src.get('vocab_size', cfg['vocab_size'])
        cfg['z_channels'] = src.get('z_channels', cfg['z_channels'])
        cfg['ch']         = src.get('ch', cfg['ch'])
        cfg['beta']       = src.get('beta', cfg['beta'])
        cfg['quant_resi'] = src.get('quant_resi', cfg['quant_resi'])
        cfg['share_quant_resi'] = src.get('share_quant_resi', cfg['share_quant_resi'])
        cfg['v_patch_nums'] = _to_tuple_int(src.get('v_patch_nums', cfg['v_patch_nums']))

    vq = VQWGAN(
        vocab_size=cfg['vocab_size'],
        z_channels=cfg['z_channels'],
        ch=cfg['ch'],
        beta=cfg['beta'],
        quant_resi=cfg['quant_resi'],
        share_quant_resi=cfg['share_quant_resi'],
        v_patch_nums=cfg['v_patch_nums'],
        test_mode=True,
    ).to(device).eval()

    # Choisir la bonne clé de poids
    if 'generator' in ckpt:
        state = ckpt['generator']
    elif 'vqwgan' in ckpt:
        state = ckpt['vqwgan']
    elif 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt

    # Filtrer D
    state = {k:v for k,v in state.items()
             if not (k.startswith('discriminator.') or k.startswith('patch_discriminator.'))}

    vq.load_state_dict(state, strict=False)
    for p in vq.parameters():
        p.requires_grad = False
    return vq

def build_var(vq, var_ckpt, depth, embed_dim, num_heads, drop_rate, attn_drop_rate, device):
    patch_nums = getattr(vq.quantize, 'v_patch_nums', (1,2,4,8))
    var = VAR(
        vae_local=vq,
        num_classes=10,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=0.1,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        patch_nums=patch_nums,
        flash_if_available=True,
        fused_if_available=True,
    ).to(device).eval()

    if not os.path.exists(var_ckpt):
        raise FileNotFoundError(var_ckpt)
    ckpt = torch.load(var_ckpt, map_location=device, weights_only=False)
    if 'model' in ckpt:
        var.load_state_dict(ckpt['model'], strict=False)
    else:
        var.load_state_dict(ckpt, strict=False)
    return var

def parse_args():
    ap = argparse.ArgumentParser("Generate class-conditional samples with VAR+VQWGAN (CIFAR-10)")
    ap.add_argument("--vqwgan_ckpt", type=str, required=True, help="Path to VQWGAN checkpoint (.pth)")
    ap.add_argument("--var_ckpt", type=str, required=True, help="Path to VAR checkpoint (.pth)")
    ap.add_argument("--class", dest="class_name_or_id", type=str, required=True,
                    help="Classe CIFAR-10 (nom ou id 0..9). Ex: 'car' ou '1'")
    ap.add_argument("--num", type=int, default=64, help="Nombre d'images à générer (multiple de 10 recommandé)")
    ap.add_argument("--cfg", type=float, default=1.5, help="Classifier-free guidance scale")
    ap.add_argument("--top_k", type=int, default=512, help="Top-k sampling")
    ap.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    ap.add_argument("--temperature", type=float, default=1.0, help="Température (si supportée par ton VAR)")
    ap.add_argument("--seed", type=int, default=0, help="Graine aléatoire")
    ap.add_argument("--outdir", type=str, default="./output/var_samples", help="Dossier de sortie")
    # paramètres “structure” du VAR (doivent correspondre à l’entraînement)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--drop_rate", type=float, default=0.0)
    ap.add_argument("--attn_drop_rate", type=float, default=0.0)
    return ap.parse_args()

def resolve_class_id(s):
    # Accepte int sous forme de string, ou nom
    if s.isdigit():
        cid = int(s)
        if not (0 <= cid <= 9):
            raise ValueError("class id must be in [0..9]")
        return cid
    s = s.lower().strip()
    if s not in NAME_TO_ID:
        raise ValueError(f"Classe inconnue '{s}'. Utilise un id (0..9) ou un nom parmi: {CIFAR10_CLASSES}")
    return NAME_TO_ID[s]

@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # charge VQWGAN + VAR
    vq = load_vqwgan(args.vqwgan_ckpt, device)
    vocab = getattr(vq, 'vocab_size', 1024)
    top_k = max(1, min(args.top_k, vocab - 1))

    var = build_var(
        vq, args.var_ckpt,
        depth=args.depth, embed_dim=args.embed_dim, num_heads=args.num_heads,
        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
        device=device
    )

    # classe
    class_id = resolve_class_id(args.class_name_or_id)
    print(f"→ Génération pour la classe {class_id} ({CIFAR10_CLASSES[class_id]})")

    # batchs de génération
    B = args.num
    labels = torch.full((B,), class_id, dtype=torch.long, device=device)

    # la méthode que tu utilises déjà dans ton code
    # si ton VAR supporte 'temperature', ajoute-le ; sinon enlève-le.
    try:
        samples = var.autoregressive_infer_cfg(
            B=B,
            label_B=labels,
            cfg=args.cfg,
            top_k=top_k,
            top_p=args.top_p,
            g_seed=args.seed,
            more_smooth=False,
            temperature=args.temperature,  # retire si non supporté
        )  # [B,3,H,W] in [0,1]
    except TypeError:
        # fallback si temperature n'est pas supporté par ta version
        samples = var.autoregressive_infer_cfg(
            B=B,
            label_B=labels,
            cfg=args.cfg,
            top_k=top_k,
            top_p=args.top_p,
            g_seed=args.seed,
            more_smooth=False,
        )

    # clamp sécurité
    samples = samples.clamp(0, 1)

    # sauvegardes
    grid_path = os.path.join(args.outdir, f"samples_{CIFAR10_CLASSES[class_id]}_N{B}_seed{args.seed}.png")
    # nrow “carré” si possible
    nrow = 10
    if int(B**0.5)**2 == B:
        nrow = int(B**0.5)
    save_image(samples, grid_path, nrow=nrow)
    print(f"✅ Grille sauvegardée : {grid_path}")

    # fichiers individuels
    indiv_dir = os.path.join(args.outdir, f"{CIFAR10_CLASSES[class_id]}_seed{args.seed}")
    os.makedirs(indiv_dir, exist_ok=True)
    for i, img in enumerate(samples):
        save_image(img, os.path.join(indiv_dir, f"{i:04d}.png"))
    print(f"✅ {B} images individuelles sauvegardées dans : {indiv_dir}")

if __name__ == "__main__":
    main()
