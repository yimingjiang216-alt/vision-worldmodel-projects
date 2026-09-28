# -*- coding: utf-8 -*-
"""过程可视化: DDIM 采样轨迹 + 数据集 + 训练曲线.

输出:
  results_process/01_dataset.png        数据集样本(动作->视频)
  results_process/02_denoising.png      从纯噪声逐步去噪的轨迹
  results_process/03_control.png        动作可控性(转向-位移曲线 + 帧序列)
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as D
from model import VideoDiT
from diffusion import Diffusion

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt8/video_dit.pt")
ap.add_argument("--out", default="results_process")
ap.add_argument("--cfg", type=float, default=8.0)
ap.add_argument("--n_ep", type=int, default=16)
ap.add_argument("--n_ctrl", type=int, default=12)
_args = ap.parse_args()
OUT = _args.out
os.makedirs(OUT, exist_ok=True)
CKPT = _args.ckpt
CFG = _args.cfg

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
a = ck["args"]
model = VideoDiT(img_size=a["size"], patch=8, in_ch=3, T=a["T"], action_dim=2,
                 dim=a["dim"], depth=a["depth"], heads=4, cond_dim=a["dim"])
model.load_state_dict(ck["model"]); model.eval()
diff = Diffusion(n_steps=a["n_steps"], device="cpu", predict=a.get("target", "eps"))
T_, sz = a["T"], a["size"]
boxes = D.build_scene(0)
ds = D.ActionVideoDataset(_args.n_ep, T_, sz, 0, boxes)

def grid(video, ncol=None):
    """(T,3,H,W) -> (T*H, W, 3) or (H, T*W, 3)."""
    x = video.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    if ncol is None:
        return x.reshape(T_ * sz, sz, 3)
    return np.concatenate([x[t] for t in range(T_)], axis=1)

# ---------- 图1: 数据集 ----------
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for k in range(6):
    vid, acts = ds[k]
    ax = axes[k // 3][k % 3]
    ax.imshow(grid(vid))
    v, om = acts[0, 0].item() * 3.0, acts[0, 1].item() * 0.9
    ax.set_title(f"v={v:.1f} m/f, omega={om:+.2f} rad/f", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Action-conditioned episodes: constant (v, omega) per episode "
             "(only actions vary -> model must use actions)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "01_dataset.png"), dpi=120)
plt.close(fig)
print("->", os.path.join(OUT, "01_dataset.png"))

# ---------- 图2: DDIM 去噪轨迹 ----------
snap_steps = [0, 6, 12, 18, 24]
torch.manual_seed(7)
acts = torch.zeros(1, T_, 2); acts[:, :, 0] = 1.0; acts[:, :, 1] = 0.2
n_sample = 25
step_idx = torch.linspace(0, a["n_steps"] - 1, n_sample, dtype=torch.long)
x = torch.randn(1, T_, 3, sz, sz)
snaps = {}
null = torch.zeros_like(acts)
for k in reversed(range(len(step_idx))):
    t = int(step_idx[k]); tt = torch.full((1,), t, dtype=torch.long)
    with torch.no_grad():
        out = model(x, tt, acts)
        if CFG != 1.0:
            out_null = model(x, tt, null)
            out = out_null + CFG * (out - out_null)
        x0, eps = diff._to_x0_eps(out, x, tt)
    if k in snap_steps:
        snaps[k] = x0.clamp(0, 1)[0].clone()
    ac_prev = diff.alphas_cumprod[int(step_idx[k - 1])] if k > 0 \
        else torch.ones_like(diff.alphas_cumprod[t])
    x = ac_prev.sqrt() * x0 + (1 - ac_prev).sqrt() * eps

fig, axes = plt.subplots(len(snap_steps), 1, figsize=(6, 2.6 * len(snap_steps)))
for ax, k in zip(axes, snap_steps):
    ax.imshow(grid(snaps[k]))
    ax.set_title(f"DDIM step {k}/{n_sample}  (t={int(step_idx[k])}, "
                 f"x0 prediction)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Generation trajectory: pure noise -> video "
             f"(target=x0, CFG={CFG:.0f})", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_denoising.png"), dpi=120)
plt.close(fig)
print("->", os.path.join(OUT, "02_denoising.png"))

# ---------- 图3: 动作可控性 ----------
import eval_control as EC
turns = EC.TURNS
flow_gen, flow_gt = {}, {}
gen_by_turn = {}
for turn in turns:
    gens, gts = [], []
    for i in range(_args.n_ctrl):
        torch.manual_seed(5000 + i)
        at = torch.zeros(1, T_, 2); at[:, :, 0] = 1.0; at[:, :, 1] = turn
        g = diff.sample_ddim(model, (1, T_, 3, sz, sz), at,
                             n_sample=n_sample, cfg=CFG)[0].clamp(0, 1)
        gens.append(EC.mean_flow_x(g))
        gts.append(EC.mean_flow_x(EC.render_episode(boxes, 1.0, turn, T_, sz, 100 + i)))
        if i == 0:
            gen_by_turn[turn] = g
    flow_gen[turn] = float(np.mean(gens)); flow_gt[turn] = float(np.mean(gts))

fig = plt.figure(figsize=(13, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6])
ax = fig.add_subplot(gs[0, 0])
ax.plot(turns, [flow_gen[t] for t in turns], "o-", label="generated")
ax.plot(turns, [flow_gt[t] for t in turns], "s--", label="ground truth")
ax.set_xlabel("turn rate (normalized)"); ax.set_ylabel("horizontal shift (px)")
ax.set_title("Action -> flow response"); ax.legend(); ax.grid(alpha=0.3)
ax2 = fig.add_subplot(gs[0, 1])
rows = [grid(gen_by_turn[t], ncol=True) for t in turns]   # (H, T*W, 3)
ax2.imshow(np.concatenate(rows, axis=0))
for r, t in enumerate(turns):
    ax2.text(T_ * sz / 2, r * sz + sz - 8, f"turn={t:+.2f}",
             color="white", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", fc="black", alpha=0.6))
ax2.set_title("Generated episodes at different turn rates (same noise)")
ax2.set_xticks([]); ax2.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_control.png"), dpi=120)
plt.close(fig)
print("->", os.path.join(OUT, "03_control.png"))
print("完成")
