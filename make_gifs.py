# -*- coding: utf-8 -*-
"""把世界模型生成的帧序列导出为动画 GIF, 直观呈现"动作 -> 视频"."""
import argparse, os
import numpy as np
import torch
from PIL import Image

import data as D
from model import VideoDiT
from diffusion import Diffusion


def load_model(ckpt, device="cpu"):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    m = VideoDiT(img_size=a["size"], patch=8, in_ch=3, T=a["T"], action_dim=2,
                 dim=a["dim"], depth=a["depth"], heads=4, cond_dim=a["dim"])
    m.load_state_dict(ck["model"]); m.eval()
    return m, a


def to_pil_video(video, scale=6):
    """(T,3,H,W) float -> list[PIL]"""
    x = video.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    T, H, W, _ = x.shape
    return [Image.fromarray((x[t] * 255).astype(np.uint8)).resize(
        (W * scale, H * scale), Image.NEAREST) for t in range(T)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt8f/video_dit.pt")
    ap.add_argument("--out", default="gifs")
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--ddim", type=int, default=25)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--scale", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, a = load_model(args.ckpt)
    diff = Diffusion(n_steps=a["n_steps"], device="cpu", predict=a.get("target", "x0"))
    T_, sz = a["T"], a["size"]
    print(f"ckpt={args.ckpt}  T={T_} size={sz} target={a.get('target')}")

    cases = [
        ("straight", 1.0, 0.0),
        ("turn_right", 1.0, 0.3),
        ("turn_left", 1.0, -0.3),
        ("slow_right", 0.4, 0.3),
    ]
    for name, v, om in cases:
        acts = torch.zeros(1, T_, 2)
        acts[:, :, 0] = v
        acts[:, :, 1] = om
        torch.manual_seed(args.seed)
        gen = diff.sample_ddim(model, (1, T_, 3, sz, sz), acts,
                               n_sample=args.ddim, cfg=args.cfg)[0].clamp(0, 1)
        frames = to_pil_video(gen, args.scale)
        fp = os.path.join(args.out, f"gen_{name}.gif")
        frames[0].save(fp, save_all=True, append_images=frames[1:],
                       duration=220, loop=0, optimize=True)
        # 真值参照
        boxes = D.build_scene(0)
        rng = np.random.default_rng(100)
        half = 38.0
        x, y, th = 0.0, 0.0, rng.uniform(0, 2 * np.pi)
        vv, oo = v * 3.0, om * 0.9
        gframes = []
        for t in range(T_):
            x = np.clip(x + vv * np.cos(th), -half, half)
            y = np.clip(y + vv * np.sin(th), -half, half)
            th = th + oo
            img, _ = D.render_frame((x, y, th), boxes, sz, sz)
            img = np.transpose(img, (1, 2, 0))   # (3,H,W) -> (H,W,3)
            gframes.append(Image.fromarray((img * 255).astype(np.uint8)).resize(
                (sz * args.scale, sz * args.scale), Image.NEAREST))
        gframes[0].save(os.path.join(args.out, f"gt_{name}.gif"),
                        save_all=True, append_images=gframes[1:],
                        duration=220, loop=0, optimize=True)
        gt_vid = np.stack([np.asarray(g.resize((sz, sz), Image.NEAREST)) / 255.0
                           for g in gframes]).transpose(0, 3, 1, 2)
        mae = float((gen - torch.from_numpy(gt_vid.astype(np.float32))).abs().mean())
        print(f"  {name}: gen vs gt(同动作) MAE = {mae:.3f}")
