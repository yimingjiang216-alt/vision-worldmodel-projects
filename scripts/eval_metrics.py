"""
Compute PSNR / SSIM / LPIPS between rendered novel views and ground-truth frames.

No GPU required. Run locally after downloading Kaggle output frames.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def psnr(gt, pred):
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def load_image(path):
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except ImportError:
        raise ImportError("Pillow not installed. Run: pip install Pillow")


def compute_ssim(gt, pred):
    """Compute SSIM using skimage if available, otherwise return NaN."""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        return ssim_fn(gt, pred, channel_axis=2, data_range=255)
    except ImportError:
        return float("nan")


def compute_lpips(gt_path, pred_path):
    """Compute LPIPS using the lpips package if available."""
    try:
        import lpips
        import torch
        loss_fn = lpips.LPIPS(net="alex").eval()
        from PIL import Image
        import torchvision.transforms as T

        def load_tensor(path):
            img = Image.open(path).convert("RGB")
            t = T.ToTensor()(img) * 2 - 1
            return t.unsqueeze(0)

        with torch.no_grad():
            return loss_fn(load_tensor(gt_path), load_tensor(pred_path)).item()
    except ImportError:
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM/LPIPS for novel views")
    parser.add_argument("--gt_dir", required=True, help="Ground-truth frames directory")
    parser.add_argument("--pred_dir", required=True, help="Rendered frames directory")
    parser.add_argument("--output", default="metrics.json", help="Output JSON")
    parser.add_argument("--pattern", default="*.png", help="Image file pattern")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    gt_files = sorted(gt_dir.glob(args.pattern))

    results = []
    psnr_list, ssim_list, lpips_list = [], [], []

    for gt_path in gt_files:
        name = gt_path.name
        pred_path = pred_dir / name
        if not pred_path.exists():
            print(f"  [skip] missing: {name}")
            continue

        gt = load_image(gt_path)
        pred = load_image(pred_path)
        if gt.shape != pred.shape:
            pred = pred[: gt.shape[0], : gt.shape[1], :]

        p_val = psnr(gt, pred)
        s_val = compute_ssim(gt, pred)
        l_val = compute_lpips(str(gt_path), str(pred_path))

        psnr_list.append(p_val)
        ssim_list.append(s_val)
        lpips_list.append(l_val)
        results.append({"frame": name, "psnr": round(p_val, 3), "ssim": round(s_val, 4), "lpips": round(l_val, 4)})

    summary = {
        "psnr_mean": round(np.mean(psnr_list), 3) if psnr_list else None,
        "psnr_std": round(np.std(psnr_list), 3) if psnr_list else None,
        "ssim_mean": round(np.mean(ssim_list), 4) if ssim_list else None,
        "lpips_mean": round(np.mean(lpips_list), 4) if lpips_list else None,
        "num_frames": len(results),
        "per_frame": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"  PSNR: {summary['psnr_mean']} ± {summary['psnr_std']} dB")
    print(f"  SSIM: {summary['ssim_mean']}")
    print(f"  LPIPS: {summary['lpips_mean']}")
    print(f"  Evaluated {summary['num_frames']} frames → {out_path}")


if __name__ == "__main__":
    main()
