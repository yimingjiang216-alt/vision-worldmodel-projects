# VideoDiT - 动作条件视频生成

从零实现的动作条件视频扩散模型（Video Diffusion Transformer）：给定逐帧动作序列（线速度 v、角速度 ω），从纯噪声生成对应的未来视频观测。

不依赖现成的视频扩散框架，模型、采样、训练、评估全部独立实现。

> **配套项目**: [mini-vla-pi0](https://github.com/yimingjiang216-alt/mini-vla-pi0) ——
> 同一套 3D 导航数据管线（`data.py` 共享）上从零训练的 π0 风格 VLA 策略
> （观测 + 语言指令 → 动作）。两者即插即用：
> VLA 输出动作 → 由共享数据管线渲染"想象未来" → 构成"决策-预演"闭环
> （见 mini-vla-pi0 的 `world_model_loop` 图与演示视频）；后续可用本项目的
> 世界模型替换解析渲染器，走向"世界模型为 VLA 批量生成训练数据"的路线。

## 一、全景：从动作到视频的链条

```
data.py        场景 + 动作序列 → 光栅化渲染 → 16 帧 48×48 真值视频 (训练目标)
   ↓
train.py       真值视频 DDPM 前向加噪 + 动作条件 (v, ω 逐帧注入)
               → 模型学 x0 预测 (15%/25% 概率丢弃动作, 训练无条件分支)
   ↓
sample.py      纯噪声 + 动作序列 → DDIM 25 步去噪 (CFG=8) → 16 帧视频
   ↓
eval_control.py  可控性检验: 相位相关测帧间位移;
                 固定噪声只改动作 vs 固定动作只改噪声 → 响应比 + 斜率对照
   ↓
结论           生成质量成立 (MAE 0.125, 帧间差接近真值);
               动作可控性未通过 (响应比 1.67x 达标, 但斜率量级与方向不对)
```

---

## 仓库内容

| 文件 | 作用 |
|---|---|
| `model.py` | VideoDiT：patchify + DiT Block（adaLN-zero）+ FiLM 动作注入 |
| `diffusion.py` | DDPM 前向加噪 / DDPM 与 DDIM 采样 / x0 预测目标 / CFG |
| `train.py` | 训练循环（x0 预测 + 动作 dropout 以训练无条件分支） |
| `sample.py` | 从纯噪声采样生成 |
| `eval_control.py` | 动作可控性评估（相位相关 + 双消融对照） |
| `export_onnx.py` | 导出单步去噪算子（opset 18） |
| `visualize_process.py` / `make_gifs.py` | 过程可视化与 GIF 生成 |
| `data.py` | 合成第一人称导航视频数据集（向量化光栅化渲染） |
| `web/` | 浏览器 demo：ONNX + onnxruntime-web，含解析渲染器作真值对照 |

结果目录：`gifs8/` `gifs16/` `gifs16b/`（各版本生成与真值 GIF）、`results_process*/`（数据集 / 去噪 / 控制过程图）、`eval_control*/control_metrics.json`（可控性指标原始输出）。

## 模型

- 视频切 patch 成 token 序列，堆叠多头自注意力 DiT Block；
- 时间步经正弦编码 + MLP，与动作一起经 adaLN-zero 调制；
- 逐帧动作经 MLP 编码后，同时做输入级零初始化 FiLM 调制；
- 采样支持 DDPM / DDIM，以及 Classifier-Free Guidance；
- 训练目标为 x0 预测。

## 结果

| 项目 | 数值 |
|---|---|
| 参数量 | 3.0 M（patch=8 / dim=192 / depth=4） |
| 分辨率 / 帧数 | 48 x 48 / 16 帧 |
| 训练轮数 | 240 epochs（ckpt16b，从 ckpt16f 续训） |
| 生成 MAE | 0.125（ckpt16b） |
| 帧间差 | 0.071（真值 0.072，CFG=8） |
| 动作响应 / 噪声比 | CFG=8：0.74x；CFG=16：1.67x |

训练目标由 ε 预测改为 x0 预测后，纯噪声生成的 MAE 由 0.42 降到 0.135。

## 实验记录（从失败到成功的调优过程）

### 短序列模型（ckpt5 → ckpt8）

| 配置 | 训练目标 | 结果 |
|---|---|---|
| ckpt5（64x64，T=8） | ε 预测 | 去噪优秀（t=10 时 x0 MAE 0.015），但**从纯噪声生成失败**：输出趋于全黑（x0 估计被 1/√ᾱₜ 放大噪声淹没），动作消融增益≈0 |
| ckpt6（32x32，T=4） | ε 预测 | 训练损失收敛 25 倍，但纯噪声生成仍失败（MAE 0.42） |
| ckpt7（32x32，T=4） | **x0 预测** | 生成成功：MAE 0.42 → **0.135**；但动作响应（0.0135）小于采样噪声（0.082），条件被稀释 |
| ckpt8（32x32，T=4） | x0 + **CFG dropout 0.15** | CFG 采样下动作响应/噪声比 0.10 → **1.14x**（cfg=8），转向-位移斜率与真值同号 |

### 长序列模型（ckpt16f / ckpt16b，16 帧 / 48x48）

4/8 帧生成结果"不像视频"（时长不足 1 秒），为此扩展序列长度与分辨率：

| 配置 | 帧数 | 分辨率 | epochs | cfg_dropout | val_mse | 生成 MAE | 响应/噪声比 |
|---|---|---|---|---|---|---|---|
| ckpt8f | 8 | 32x32 | 150 | 0.15 | 0.0036 | 0.166 | 1.14x（CFG=8） |
| ckpt16f | 16 | 48x48 | 90 | 0.15 | 0.0047 | 0.136 | 0.24x（CFG=8） |
| **ckpt16b** | **16** | **48x48** | **240** | **0.25** | **0.0030** | **0.125** | **1.67x（CFG=16，通过 >1.5x 判据）** |

关键改进：把 CFG dropout 由 0.15 提高到 0.25 并续训至 240 epochs——更充分的无条件分支训练让采样时 CFG 有更大可放大余量（CFG 扫描：8→0.74x，16→1.67x，24→2.69x，32→3.75x，单调放大）。代价是高 CFG 下画面动态会被抑制，因此展示用 CFG=8（动态自然），可控性评估用 CFG=16（响应充分）。

## 动作可控性：未通过

`eval_control.py` 的判据是两条同时满足：

```python
ratio_ok = ratio > 1.5
slope_ok = (slope_gen * slope_gt > 0) and (abs(slope_gen) > 0.5 * abs(slope_gt))
controllable = ratio_ok and slope_ok
```

三个版本的实际输出（见 `eval_control*/control_metrics.json`）：

| 版本 | 响应/噪声比 | ratio_ok | 生成斜率 | 真值斜率 | slope_ok | action_controllable |
|---|---|---|---|---|---|---|
| ckpt8 | 1.138 | false | +0.968 | +16.684 | false | **false** |
| ckpt16f | 0.647 | false | -0.218 | +18.371 | false | **false** |
| ckpt16b | 1.669 | **true** | -4.038 | +18.371 | false | **false** |

ckpt16b 通过了响应比这一条（1.67x > 1.5），但转向-位移斜率量级只有真值的约 22%，且符号与真值相反。因此脚本最终判定 **动作可控性未通过**。

结论：模型确实对动作产生了响应（改动作会改变输出），但响应的方向与强度都不正确，不足以支撑闭环控制。

## 评估方法

没有用现成的视频质量指标，而是自己设计了判据：

- 用相位相关估计帧间位移；
- 先在真值渲染视频上校准指标有效区间（|ω| > 0.3 rad/帧 时相位相关假设失效）；
- 做两组对照：「固定噪声、只改动作」与「固定动作、只改噪声」，以两者比值作为动作响应判据。

指标先在真值上校准，再用于生成结果。

## 方法与出处

| 本仓库用到的方法 | 出处 |
|---|---|
| DiT 架构 / adaLN-zero | Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023, arXiv:2212.09748 |
| DDPM 前向加噪 / x0 预测 | Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020, arXiv:2006.11239 |
| DDIM 采样 | Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021, arXiv:2010.02502 |
| 视频扩散（时序 token 谱系） | Ho et al., *Video Diffusion Models*, arXiv:2204.03458 |
| 「生成模型当物理模拟器」的定位 | 思想溯源至 NVIDIA, *Cosmos World Foundation Model Platform for Physical AI*, arXiv:2501.03575；本仓库为独立从零实现，3M 参数 / 合成小场景，与 Cosmos 的规模和数据域不同 |

## 复现

```bash
pip install -r requirements.txt
# 16 帧 / 48x48 主模型（x0 目标，约 2h CPU）
python train.py --epochs 90 --batch 64 --size 48 --T 16 --dim 192 --depth 4 \
  --target x0 --cfg_dropout 0.15 --n_train 768 --n_val 96 --out ckpt16f
# 续训加强动作可控性（cfg_dropout 0.25，再约 3.3h CPU）→ ckpt16b
python train.py --resume ckpt16f/video_dit.pt --epochs 150 --batch 64 --size 48 --T 16 \
  --dim 192 --depth 4 --target x0 --cfg_dropout 0.25 --lr 7e-4 \
  --n_train 768 --n_val 96 --out ckpt16b
# 采样 + 生成质量评估
python sample.py --ckpt ckpt16b/video_dit.pt --n 4 --ddim 25 --cfg 8
# 动作可控性严格评估（真值校准，CFG=16）
python eval_control.py --ckpt ckpt16b/video_dit.pt --n 16 --ddim 25 --cfg 16
```

Kaggle T4 GPU 一键复现（8M 参数加强版，dim 256 / depth 6，300 epochs）：
`kaggle_world_model_gpu.ipynb`，全部源码已内联，开箱即跑。

## 浏览器 demo

模型导出为内联权重的单文件 ONNX（12.5 MB），用 onnxruntime-web 在浏览器里跑完整 DDIM 采样，无需后端、无需 GPU。单次生成（DDIM 25 步 + CFG=8，16 帧 48x48）在 WASM 中约 14 秒。

页面内置一个用 JavaScript 移植的解析渲染器，可对同一动作渲染真值参考视频，与生成结果直接对照。

---

## 相关说明

扩散模型的时间步嵌入、CFG 公式、adaLN-zero 的实现细节，以及「为什么换成 x0 预测有效」「为什么 dropout 提高后响应比上升」的分析，写在配套的项目报告里。
