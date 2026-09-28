# -*- coding: utf-8 -*-
"""DDPM/DDIM 采样器 (纯 PyTorch 实现).

支持两种训练目标:
  * "eps": 预测噪声 (经典 DDPM), 高噪声时 x0 估计误差被 1/sqrt(a_t) 放大
  * "x0" : 直接预测干净样本, 对小模型/低算力更友好, 动作条件在所有噪声
           等级都直接受监督 -> 更适合 Action-conditioned World Model 的
           可控性验证.
"""
import math
import torch


class Diffusion:
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02,
                 device="cpu", predict="eps"):
        self.n_steps = n_steps
        self.device = device
        self.predict = predict
        betas = torch.linspace(beta_start, beta_end, n_steps,
                               dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas

    def q_sample(self, x0, t, noise=None):
        """前向加噪: x_t = sqrt(a_t) x0 + sqrt(1-a_t) eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        ac = self.alphas_cumprod[t]
        ac = ac.view(-1, *([1] * (x0.dim() - 1)))
        return ac.sqrt() * x0 + (1 - ac).sqrt() * noise, noise

    def _to_x0_eps(self, out, x_t, t):
        """把模型输出统一转换为 (x0, eps)."""
        ac = self.alphas_cumprod[t]
        ac = ac.view(-1, *([1] * (out.dim() - 1)))
        if self.predict == "x0":
            x0 = out
            eps = (x_t - ac.sqrt() * x0) / (1 - ac).sqrt().clamp(min=1e-8)
        else:
            eps = out
            x0 = (x_t - (1 - ac).sqrt() * eps) / ac.sqrt().clamp(min=1e-8)
        return x0, eps

    @torch.no_grad()
    def sample_ddpm(self, model, shape, actions, generator=None):
        x = torch.randn(shape, device=self.device, generator=generator)
        model.eval()
        for t in reversed(range(self.n_steps)):
            tt = torch.full((shape[0],), t, device=self.device,
                            dtype=torch.long)
            x0, eps = self._to_x0_eps(model(x, tt, actions), x, tt)
            if t > 0:
                ac_prev = self.alphas_cumprod[t - 1]
                alpha = self.alphas[t]
                mean = (alpha.sqrt() * x0 + (1 - ac_prev).sqrt() * eps)
                var = (1 - ac_prev) * (1 - alpha) / (1 - self.alphas_cumprod[t])
                noise = torch.randn(x.shape, device=self.device,
                                    generator=generator)
                x = mean + var.sqrt() * noise
            else:
                x = x0
        return x

    @torch.no_grad()
    def sample_ddim(self, model, shape, actions, n_sample=50,
                    generator=None, cfg=1.0, null_actions=None):
        """DDIM 采样. cfg>1 时启用 Classifier-Free Guidance:
        out = out_uncond + cfg * (out_cond - out_uncond), 放大动作条件.
        null_actions: 无条件分支的输入动作(一般为全零)."""
        model.eval()
        step_idx = torch.linspace(0, self.n_steps - 1, n_sample,
                                  dtype=torch.long)
        x = torch.randn(shape, device=self.device, generator=generator)
        if cfg != 1.0 and null_actions is None:
            null_actions = torch.zeros_like(actions)
        for k in reversed(range(len(step_idx))):
            t = int(step_idx[k])
            tt = torch.full((shape[0],), t, device=self.device,
                            dtype=torch.long)
            out = model(x, tt, actions)
            if cfg != 1.0:
                out_null = model(x, tt, null_actions)
                out = out_null + cfg * (out - out_null)
            x0, eps = self._to_x0_eps(out, x, tt)
            if k > 0:
                ac_prev = self.alphas_cumprod[int(step_idx[k - 1])]
            else:
                ac_prev = torch.ones_like(self.alphas_cumprod[t])
            x = ac_prev.sqrt() * x0 + (1 - ac_prev).sqrt() * eps
        return x
