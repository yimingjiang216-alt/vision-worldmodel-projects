# -*- coding: utf-8 -*-
"""动作可控性专项评估 (严格版).

三重判据:
  A. 响应/噪声比: 固定噪声改变转向 vs 固定转向改变噪声, 前者应显著更大
  B. 转向-位移斜率: 生成视频的水平位移应随转向角速度单调变化, 且斜率
     符号与量级与真值渲染一致 (校准过的物理量, 非任意阈值)
  C. 真值参照: 同一动作集下真值渲染的位移统计

注: 转向扫描限制在 |om|<=0.3 内 —— 先验实验(_gt_flow.py)表明超出该区间
后帧间旋转过大, 相位相关位移估计在真值上也不再单调, 指标失效.
"""
import argparse, json, os
import numpy as np
import torch

import data as D
from model import VideoDiT
from diffusion import Diffusion

TURNS = [-0.3, -0.15, 0.0, 0.15, 0.3]
V_NORM = 1.0


def load_model(ckpt_path, device="auto"):
    import torch as _T
    if device == "auto":
        device = "cuda" if _T.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    model = VideoDiT(img_size=a["size"], patch=8, in_ch=3, T=a["T"],
                     action_dim=2, dim=a["dim"], depth=a["depth"],
                     heads=4, cond_dim=a["dim"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, a


def subpixel_shift_x(a_, b_):
    A = np.fft.rfft2(a_); B = np.fft.rfft2(b_)
    R = A * np.conj(B); R /= (np.abs(R) + 1e-8)
    cc = np.fft.irfft2(R, s=a_.shape)
    W = a_.shape[1]; idx = np.argmax(cc); s0 = idx % W
    if s0 > W / 2: s0 -= W
    def at(s):
        s = int(round(s)) % W
        return cc[0, s]
    ym, y0, yp = at(s0 - 1), at(s0), at(s0 + 1)
    denom = (ym - 2 * y0 + yp)
    delta = 0.5 * (ym - yp) / denom if abs(denom) > 1e-12 else 0.0
    return s0 + delta


def mean_flow_x(video):
    """相邻帧相对前一帧的水平位移均值 (px)."""
    gray = video.mean(dim=1).numpy()
    return float(np.mean([subpixel_shift_x(gray[t], gray[t + 1])
                          for t in range(gray.shape[0] - 1)]))


def render_episode(boxes, v_norm, om_norm, T, size, seed):
    rng = np.random.default_rng(seed)
    half = 38.0
    x, y, th = 0.0, 0.0, rng.uniform(0, D.TWO_PI)
    v = v_norm * 3.0; om = om_norm * 0.9
    frames = []
    for t in range(T):
        x = np.clip(x + v * np.cos(th), -half, half)
        y = np.clip(y + v * np.sin(th), -half, half)
        th = th + om
        img, dep = D.render_frame((x, y, th), boxes, size, size)
        frames.append(img)
    return torch.from_numpy(np.stack(frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt8/video_dit.pt")
    ap.add_argument("--out", default="eval_control")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--ddim", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("评估设备:", dev)
    model, a = load_model(args.ckpt, dev)
    predict = a.get("target", "eps")
    diff = Diffusion(n_steps=a["n_steps"], device=dev, predict=predict)
    print(f"ckpt={args.ckpt} target={predict} T={a['T']} size={a['size']} cfg={args.cfg}")

    boxes = D.build_scene(0)
    T_, sz = a["T"], a["size"]

    def gen(turn, seed):
        acts = torch.zeros(1, T_, 2)
        acts[:, :, 0] = V_NORM; acts[:, :, 1] = turn
        shape = (1, T_, 3, sz, sz)
        torch.manual_seed(seed)
        return diff.sample_ddim(model, shape, acts, n_sample=args.ddim,
                                cfg=args.cfg)[0].clamp(0, 1)

    gen_by_turn = {t: [] for t in TURNS}
    noise_baseline = []
    gt_by_turn = {t: [] for t in TURNS}
    for i in range(args.n):
        for turn in TURNS:
            gen_by_turn[turn].append(gen(turn, 5000 + i))
            gt_by_turn[turn].append(render_episode(boxes, V_NORM, turn, T_, sz, 100 + i))
        noise_baseline.append(gen(0.0, 90000 + i))   # 同动作(直行), 换噪声

    resp = [float((gen_by_turn[-0.3][i] - gen_by_turn[0.3][i]).abs().mean())
            for i in range(args.n)]
    base = [float((gen_by_turn[0.0][i] - noise_baseline[i]).abs().mean())
            for i in range(args.n)]
    m_resp, m_base = float(np.mean(resp)), float(np.mean(base))
    ratio = m_resp / max(m_base, 1e-8)

    flow_gen = {t: float(np.mean([mean_flow_x(g) for g in gen_by_turn[t]]))
                for t in TURNS}
    flow_gt = {t: float(np.mean([mean_flow_x(g) for g in gt_by_turn[t]]))
               for t in TURNS}

    x_ = np.array(TURNS, dtype=float)
    slope_gen = float(np.polyfit(x_, [flow_gen[t] for t in TURNS], 1)[0])
    slope_gt = float(np.polyfit(x_, [flow_gt[t] for t in TURNS], 1)[0])

    print(f"  [A] 动作响应 |gen(-.3)-gen(+.3)| = {m_resp:.4f}")
    print(f"      噪声基线 |gen(z1)-gen(z2)|   = {m_base:.4f}")
    print(f"      响应/噪声比                 = {ratio:.2f}x")
    print(f"  [B] 生成 flow: " +
          "  ".join(f"{t:+.2f}:{flow_gen[t]:+.2f}" for t in TURNS))
    print(f"      真值 flow: " +
          "  ".join(f"{t:+.2f}:{flow_gt[t]:+.2f}" for t in TURNS))
    print(f"      斜率 dflow/dturn  生成={slope_gen:+.2f}  真值={slope_gt:+.2f} px/(rad/s_norm)")

    ratio_ok = ratio > 1.5
    slope_ok = (slope_gen * slope_gt > 0) and (abs(slope_gen) > 0.5 * abs(slope_gt))
    controllable = ratio_ok and slope_ok

    summary = dict(
        ckpt=args.ckpt, target=predict, n=args.n, ddim=args.ddim, cfg=args.cfg,
        action_response_mae=m_resp, noise_baseline_mae=m_base,
        response_noise_ratio=ratio,
        flow_gen={str(t): flow_gen[t] for t in TURNS},
        flow_gt={str(t): flow_gt[t] for t in TURNS},
        slope_gen=slope_gen, slope_gt=slope_gt,
        ratio_ok=bool(ratio_ok), slope_ok=bool(slope_ok),
        action_controllable=bool(controllable),
    )
    with open(os.path.join(args.out, "control_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n汇总:", json.dumps(summary, ensure_ascii=False, indent=2))
    print("动作可控性:", "通过" if controllable else "未通过",
          "(响应>1.5x噪声 且 生成斜率与真值同号且>=50%量级)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        axes[0].plot(TURNS, [flow_gen[t] for t in TURNS], "o-", label="generated")
        axes[0].plot(TURNS, [flow_gt[t] for t in TURNS], "s--", label="ground truth")
        axes[0].set_xlabel("turn rate (normalized)")
        axes[0].set_ylabel("mean horizontal shift (px)")
        axes[0].set_title(f"action->flow  slope gen={slope_gen:+.2f} gt={slope_gt:+.2f}")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        i0 = 0
        for ax, t in zip(axes[1:], [TURNS[0]]):
            pass
        # 帧序列拼图
        fig2, axes2 = plt.subplots(len(TURNS), 1, figsize=(8, 9))
        for ax, t in zip(axes2, TURNS):
            v = gen_by_turn[t][i0].permute(0, 2, 3, 1).reshape(
                T_ * sz, sz, 3).numpy()
            ax.imshow(v); ax.axis("off")
            ax.set_title(f"turn={t:+.2f}  flow={flow_gen[t]:+.2f}px (gt {flow_gt[t]:+.2f})")
        fig2.tight_layout()
        fig2.savefig(os.path.join(args.out, "turn_sweep.png"), dpi=110)
        plt.close(fig2)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "flow_curve.png"), dpi=110)
        plt.close(fig)
        print("已保存 turn_sweep.png / flow_curve.png")
    except Exception as e:
        print("绘图跳过:", e)


if __name__ == "__main__":
    main()
