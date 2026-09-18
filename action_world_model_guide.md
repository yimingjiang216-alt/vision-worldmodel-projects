# 项目 C：Action-conditioned 视频世界模型 —— 从零到跑通的实操脚本

> 目标：**输入当前帧 + 动作（或相机位姿）→ 预测未来帧 / 未来轨迹**，构建一个最小的"动作条件世界模型"。
> 对应岗位关键词：**World Model、Action-conditioned、Spatial-aware、Scene Completion、VLA 数据生成、Embodied AI**。
> 前置：项目 A/B 是它的素材与条件来源。预计：环境 0.5-1 天 + 小数据训练 1-2 天。

---

## 第 0 步：先理解"世界模型"在这个岗指什么
岗位说：**研究 3D-aware、Spatial-aware、Action-conditioned World Model；探索其在仿真、VLA 数据生成、场景补全与新视角生成中的应用。**
- **World Model**：学习环境动力学，能"想象/预测"未来。
- **Action-conditioned**：把"动作/控制"作为条件，预测接下来的状态/画面——这是与纯视频生成的最大区别。
- 用它做 **仿真 / VLA（Vision-Language-Action）训练数据生成**：生成可控、可标注的动作-观测序列，喂给机器人策略。

---

## 第 1 步：选方案（两条路线，建议先跑通路线一）

### 路线一：复现经典 world model（最稳、最可复现）
- **DreamerV3**（Hafner, 深度强化学习 world model + 想象力策略）
- 体验核心：**观测 → 学习世界动力学 → 在 latent 空间"想象"未来 → 训练策略**。
- 数据集/环境：**DMControl**（连续控制）、**Atari** 等。

### 路线二：自建最小 action-conditioned 视频预测（更贴合视频生成岗）
- 数据：**BAIR robot pushing** / **RoboNet**（机器人短视频 + 动作）或**自采/合成**数据。
- 模型：小 VAE/特征提取 + **动作条件**的时序/扩散预测器（latent video diffusion 或 Transformer）。
- 评估：预测下一帧 vs 真值（PSNR/SSIM/FVD）；**动作一致性**（两个不同动作 → 预测应不同）。

> 建议：先跑通**路线一**建立"世界模型"概念，再把**路线二**作为简历主项目（因为它直接对应"视频生成 + 动作条件"）。

---

## 第 2 步：路线一 —— DreamerV3 复现

```bash
git clone https://github.com/danijar/dreamerv3.git
cd dreamerv3
pip install -r requirements.txt
# 在 DMControl 上训练 world model + 策略
python dreamerv3/train.py --logdir ~/logdir/dmc_wm --env dmc_hopper_hop
# 用 tensorboard 看训练曲线；训练完成后回放/可视化想象轨迹
```
> 输出：**world model 的训练曲线**、在环境中的策略回报、**latent 空间想象未来**的可视化。

---

## 第 3 步：路线二 —— 自建 action-conditioned 视频预测（核心项目）

### 3.1 数据处理
- 下载 **BAIR robot pushing** 或 **RoboNet** 子集；或**采集自己的数据**（如用一个物体做推动/旋转，记录短视频 + 动作标签）。
- 构造：(观测帧 t，动作 a) → 未来帧 t+1..t+k。建议先用小规模（几百-几千条）跑通，再扩。

### 3.2 模型设计（最小可行）
```
输入：当前帧 x_t（或某 latent z_t） + 动作 a
流程：编码器 → 条件动力学 f(z_t, a) → 解码器 → 预测帧 x_{t+1}
可选用：小 VAE / 低维特征 + 条件扩散（latent diffusion conditioned on action）
      或 Transformer 自回归逐帧预测（video GPT 思路）
```
- 关键点：**动作如何注入**（concat / cross-attention / FiLM），要能证明"动作真的在影响未来"。

### 3.3 训练
```bash
python train_wm.py --data <数据> --horizon 5 --cond action --model latent_diffusion
# 监控：训练损失 / 预测 PSNR / SSIM
```

### 3.4 推理与可视化
- 给定初始帧 + 一串动作，**推出未来若干帧 / 轨迹**，渲染成视频。
- 做**动作对照实验**：同一初始帧、两串不同动作 → 生成两条不同未来，证明 action-conditioned。

---

## 第 4 步：量化评测（写进简历）
- **预测精度**：与真值未来帧算 **PSNR / SSIM / LPIPS / FVD**。
- **动作一致性**：不同动作预测应显著不同（计算动作-预测相关性、或分类器能区分）。
- **长程稳定性**：多步自回归预测是否漂移 / 崩坏（记录 horizon 内的误差累积）。
- 与 **无动作条件的纯视频预测** 对比，突出"动作条件带来可控性提升"。

---

## 第 5 步：常见问题 & 调参

| 问题 | 解决 |
|---|---|
| 预测很快模糊 / 崩坏 | 减小预测 horizon、用 latent（而非像素）预测、加 VAE 约束 |
| 动作对结果"没影响" | 检查动作注入方式与归一化；做消融（有无动作）对比 |
| 训练不收敛 | 数据量小先过拟合一个小 batch，确认链路无误再放大 |
| 长程漂移 | 用 autoregressive 但加噪声/teacher forcing 缓解；或换 latent world model |
| 想接 VLA | 参考 Open X-Embodiment / RT-2 / π0，用 world model 生成"观测-动作"序列当数据 |

---

## 第 6 步：写成简历条目（对应 World Model / Action-conditioned / VLA）

> 参考文案（**真实跑通后写**）：
>
> **Action-conditioned 视频世界模型    2026.xx**
> - 构建最小可复现的**动作条件世界模型**：输入当前帧与动作，预测未来帧 / 未来轨迹，实现环境动力学建模与"想象未来"；
> - 基于 [DreamerV3 / 自建 latent video diffusion] 实现，理解 latent world model、动作条件注入（cross-attention / FiLM）与自回归预测；
> - 在机器人/合成视频数据集上以 PSNR / SSIM / LPIPS / FVD 定量评测，并做**动作对照实验**验证动作条件的可控性（同一初始帧、不同动作 → 不同未来）；
> - 探索将其用于**仿真与 VLA 训练数据生成、场景补全与多步轨迹预测**，服务于行动条件的世界模型与具身智能方向。

**面试一句话**：
> "我复现并构建了 action-conditioned 视频世界模型，能根据动作预测未来帧，并定量验证了动作对预测的可控性，这可以用在仿真和 VLA 数据生成上。"

---

## 附：与岗位的衔接
- 直接对应"**研究 3D-aware、Spatial-aware、Action-conditioned World Model**"与"**探索世界模型在仿真、VLA 数据生成、场景补全与新视角生成中的应用**"。
- 若把项目 A 的 **3DGS 位姿/深度**和项目 B 的**可控视频**接入，可作为世界模型的**场景条件与训练数据**，形成 A→B→C 完整闭环，面试叙事非常有说服力。
