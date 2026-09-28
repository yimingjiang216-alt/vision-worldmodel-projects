# -*- coding: utf-8 -*-
"""Video DiT (Diffusion Transformer) 世界模型.

架构要点:
  * Patchify: 视频 (B,T,3,H,W) -> 非重叠 patch -> token 序列
  * DiT Block: LayerNorm + 多头自注意力 + MLP, 时间步与动作条件通过
    adaLN-zero 调制(与 DiT/Sora 同一思想)注入
  * 时间步条件: 正弦位置编码 -> MLP
  * 动作条件: 每帧动作 a_t -> MLP, 与时间步相加, 实现 action-conditioned 生成
  * 训练目标: epsilon-prediction (DDPM)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(t, dim, max_period=10000.0):
    """扩散时间步的正弦位置编码 (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Mlp(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DiTBlock(nn.Module):
    """DiT block with adaLN-zero conditioning.

    调制参数由条件向量 c 预测: (scale_msa, shift_msa, gate_msa,
    scale_mlp, shift_mlp, gate_mlp), 初始化为零 -> 训练初期等价于恒等映射.
    """

    def __init__(self, dim, heads=4, mlp_ratio=4.0, cond_dim=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden, dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        B, L, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=-1)
        # c 已是 (B, L, cond), 调制参数逐 token, 直接按位相乘
        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa * h
        h = self.norm2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        return x + gate_mlp * h


class VideoDiT(nn.Module):
    """Action-conditioned Video Diffusion Transformer.

    输入 video (B,T,3,H,W), actions (B,T,A), timestep (B,)
    输出预测噪声 (B,T,3,H,W).
    """

    def __init__(self, img_size=64, patch=8, in_ch=3, T=8, action_dim=2,
                 dim=256, depth=6, heads=4, cond_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch = patch
        self.T = T
        self.dim = dim
        self.action_dim = action_dim
        self.n_tokens = (img_size // patch) ** 2 * T
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch,
                                     stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # 时间步条件
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim))
        # 动作条件: 每帧动作 -> 嵌入, 与时间步融合
        self.act_mlp = nn.Sequential(
            nn.Linear(action_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim))
        # 每帧动作的广播位置: token 需要知道自己属于哪一帧
        self.frame_embed = nn.Parameter(torch.zeros(1, T, cond_dim))
        nn.init.trunc_normal_(self.frame_embed, std=0.02)
        # 输入级 FiLM 调制: 动作直接注入 token 特征(不依赖 adaLN 门控),
        # 保证动作条件在训练初期就有强梯度, 避免"动作被忽略"
        self.act_film = nn.Sequential(
            nn.Linear(action_dim, dim), nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.act_film[-1].weight)
        nn.init.zeros_(self.act_film[-1].bias)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads, cond_dim=cond_dim) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.out_proj = nn.Linear(dim, patch * patch * in_ch)

    def _tokens(self, video):
        """(B,T,3,H,W) -> (B, L, D) token 序列, 帧顺序排列."""
        B, T, C, H, W = video.shape
        x = video.reshape(B * T, C, H, W)
        x = self.patch_embed(x)                    # (B*T, D, h, w)
        x = x.flatten(2).transpose(1, 2)           # (B*T, L_t, D)
        x = x.reshape(B, T * x.shape[1], x.shape[2])
        return x

    def _cond(self, t, actions):
        """融合时间步 + 动作序列 -> 每帧条件向量 (B, T, cond_dim)."""
        temb = sinusoidal_timestep_embedding(t, self.frame_embed.shape[2])
        temb = self.t_mlp(temb)                    # (B, cond)
        aemb = self.act_mlp(actions)               # (B, T, cond)
        c = aemb + temb[:, None] + self.frame_embed
        return c                                   # (B, T, cond)

    def forward(self, video, t, actions):
        """video:(B,T,3,H,W) t:(B,) actions:(B,T,A) -> noise (B,T,3,H,W)."""
        B, T, C, H, W = video.shape
        x = self._tokens(video)
        x = x + self.pos_embed[:, :x.shape[1]]
        # 输入级 FiLM: 每帧动作 -> (scale, shift) 调制该帧所有 token
        film = self.act_film(actions)              # (B, T, 2*dim)
        L_t = x.shape[1] // self.T
        film = film.repeat_interleave(L_t, dim=1)  # (B, L, 2*dim)
        scale_f, shift_f = film.chunk(2, dim=-1)
        x = x * (1 + scale_f) + shift_f
        c = self._cond(t, actions)                 # (B, T, cond)
        c_rep = c.repeat_interleave(L_t, dim=1)    # (B, L, cond)
        for blk in self.blocks:
            x = blk(x, c_rep)
        shift, scale = self.final_adaLN(c_rep).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.out_proj(x)                       # (B, L, patch*patch*C)
        p = self.patch
        h = w = H // p
        x = x.reshape(B, T, h, w, p, p, C)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, T, C, H, W)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)