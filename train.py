# -*- coding: utf-8 -*-
"""训练脚本: 在动作条件视频数据集上训练 Video DiT 世界模型.

用法:
    python train.py --epochs 30 --batch 16
"""
import argparse
import os
import time
import numpy as np
import torch

import data as D
from model import VideoDiT, count_params
from diffusion import Diffusion


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs_gpu", type=int, default=0,
                    help="GPU 加速档: >0 时覆盖 epochs(如 Kaggle T4 用 600)")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n_steps", type=int, default=250)
    ap.add_argument("--cfg_dropout", type=float, default=0.0,
                    help="Classifier-Free Guidance: 以该概率丢弃动作(置零), "
                         "训练无条件分支, 采样时用 --cfg 放大动作条件")
    ap.add_argument("--target", type=str, default="eps",
                    choices=["eps", "x0"],
                    help="扩散预测目标: eps=预测噪声, x0=直接预测干净样本")
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_val", type=int, default=32)
    ap.add_argument("--out", type=str, default="ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto",
                    help="cpu / cuda / auto: Kaggle T4 GPU 自动启用")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--resume", type=str, default="",
                    help="从该 checkpoint 继续训练(复用已训权重, 重新初始化优化器)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.epochs_gpu > 0:
        args.epochs = args.epochs_gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    print(f"设备: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    tr_loader, va_loader = D.make_loaders(
        n_train=args.n_train, n_val=args.n_val, T=args.T, size=args.size,
        batch=args.batch, seed=args.seed)

    model = VideoDiT(img_size=args.size, patch=8, in_ch=3, T=args.T,
                     action_dim=2, dim=args.dim, depth=args.depth,
                     heads=4, cond_dim=args.dim).to(dev)
    start_ep = 0
    if args.resume:
        import torch as _T
        ck = _T.load(args.resume, map_location=dev, weights_only=False)
        ra = ck["args"]
        assert ra["size"] == args.size and ra["T"] == args.T, \
            "resume 配置不匹配: size/T 与当前参数不一致"
        model.load_state_dict(ck["model"])
        start_ep = ra.get("epochs", 0)
        print(f"从 {args.resume} 恢复 (已训 {start_ep} epochs, "
              f"沿用其权重, 优化器重新初始化)")
    print(f"模型参数量: {count_params(model)/1e6:.2f}M  目标: {args.target} "
          f"cfg_dropout={args.cfg_dropout}")
    diff = Diffusion(n_steps=args.n_steps, device=dev, predict=args.target)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def run_epoch(loader, train=True):
        model.train(train)
        tot, nb = 0.0, 0
        for video, acts in loader:
            video = video.to(dev); acts = acts.to(dev)
            B = video.shape[0]
            t = torch.randint(0, diff.n_steps, (B,), device=dev)
            x_t, eps = diff.q_sample(video, t)
            # CFG: 随机丢弃部分样本的动作 -> 训练无条件分支
            if args.cfg_dropout > 0:
                keep = (torch.rand(B, device=dev) >= args.cfg_dropout)
                acts_in = acts * keep[:, None, None].float()
            else:
                acts_in = acts
            pred = model(x_t, t, acts_in)
            if args.target == "x0":
                loss = torch.nn.functional.mse_loss(pred, video)
            else:
                loss = torch.nn.functional.mse_loss(pred, eps)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item(); nb += 1
        return tot / max(nb, 1)

    step = 0
    t0 = time.time()
    for ep in range(start_ep, start_ep + args.epochs):
        tr_loss = run_epoch(tr_loader, train=True)
        with torch.no_grad():
            va_loss = run_epoch(va_loader, train=False)
        step += 1
        print(f"epoch {ep+1}/{start_ep + args.epochs}  train_mse={tr_loss:.5f}  "
              f"val_mse={va_loss:.5f}  elapsed={time.time()-t0:.0f}s")
    save_args = dict(vars(args))
    save_args["epochs"] = start_ep + args.epochs   # 记录累计 epoch 数
    save_args.pop("resume", None)
    torch.save({"model": model.state_dict(), "args": save_args},
               os.path.join(args.out, "video_dit.pt"))
    print("checkpoint saved:", os.path.join(args.out, "video_dit.pt"))


if __name__ == "__main__":
    main()