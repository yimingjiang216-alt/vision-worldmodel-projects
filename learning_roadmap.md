# 世界模型 / 视频生成 / 3D Vision：补齐短板学习路线 + 简历可落地项目

> 目标岗位：World Model / Video Generation / 空间智能 研发实习生
> 该岗位硬性关键词：Video Diffusion / DiT、Depth / Pose / Point Cloud / 3DGS、
> 3D-aware / Spatial-aware、Action-conditioned World Model、VLA、新视角生成、场景补全、PyTorch、论文阅读与代码复现。

---

## 一、现状与差距（你的底子 vs 岗位要求）

### ✅ 你已经具备（直接可写进简历、也是最大优势）
- SfM / MVS 三维重建全流程（特征匹配 → BA → 稠密点云 → 网格 → 纹理）
- 特征提取/匹配（SIFT、Harris、Canny、LoG）、相机位姿估计、相机标定
- 多视图几何、核线几何、坐标变换、像素级几何校正
- 点云 / DEM / DSM / 三维数据格式（OSGB、GeoTIFF）
- Python + 深度学习模型工程化（SAM）、面向对象封装、AI Agent 开发

### ❌ 与岗位的主要差距
| 短板 | 岗位要求 | 影响 |
|---|---|---|
| 未系统接触 **PyTorch** | 熟悉 Python / PyTorch | 硬门槛，必须补 |
| 未做过 **Diffusion / 视频生成** | Video Diffusion / DiT | 核心方向，必须补 |
| 未接触 **NeRF / 3DGS 实操** | 3DGS、新视角生成 | 重点加分项 |
| 未接触 **World Model / VLA** | World Model、VLA 数据生成 | 差异化亮点 |
| **论文阅读 / 复现** 未体现 | 有较强论文阅读、代码复现能力 | 硬要求，需产出物 |

---

## 二、学习路线（按 12 周规划，假设每天 3-4 小时）

### 阶段 0：PyTorch + 深度学 π 基础（第 1-2 周）
**目标：能独立写训练/推理脚本、看懂论文代码。**
1. PyTorch 官方教程（60min blitz + DataLoader + nn.Module + GPU 训练）
2. 自己手写一个 **MLP / CNN** 在 CIFAR-10 上训练到 >85%
3. 吃透 **反向传播** 推导（Karpathy《Neural Networks: Zero to Hero》）
4. 读一遍 **Transformer 结构与注意力**（Karpathy nanoGPT / Attention is All You Need）
- ✅ **产出**：一个自己的 PyTorch 训练脚本仓库（含数据加载、训练、评估、可视化）

### 阶段 1：扩散模型 Diffusion（第 3-4 周）
**目标：理解 DDPM 原理并能复现。**
1. 读论文：**DDPM（Ho et al. 2020）→ DDIM → Classifier-Free Guidance → Latent Diffusion（Stable Diffusion）**
2. 复现：用 GitHub 上 **lucidrains/denoising-diffusion-pytorch**，在 MNIST/CIFAR 上从零跑通
3. 手推 **前向加噪 / 反向去噪 / noise schedule / ε-prediction** 公式
4. 进阶：理解 **UNet 时间条件嵌入、cross-attention 文本条件、Latent Space VAE**
- ✅ **产出**：一个可生成图像的 Diffusion 脚本 + README + 生成结果图

### 阶段 2：视频生成与 DiT（第 5-6 周）
**目标：从图像扩散迁移到视频/时空扩散。**
1. 读：**Video Diffusion Models（Ho et al.）→ DiT（Diffusion Transformers，2023）→ Stable Video Diffusion→ VideoCrafter / AnimateDiff / CogVideoX / Open-Sora / Wan**
2. 复现：在公开视频扩散开源项目上跑通（**AnimateDiff / Open-Sora / Stable Video Diffusion**），理解
   - 3D 时空 attention（spatial + temporal blocks）
   - 光流 / 运动条件
   - 帧间一致性、文生视频 / 图生视频
3. 关键进阶：**把 3D 条件（Depth / Camera Pose / Point Cloud）接入视频生成**——这正是岗位"视频生成与 Depth/Pose/3DGS 结合"的核心
- ✅ **产出**：一行命令出视频的 demo + 论文笔记 + 一张"框架对比表"

### 阶段 3：3D 表示与神经渲染（第 7-8 周）
**目标：NeRF → 3DGS，并做出"新视角生成"demo。**
1. 读：**NeRF（ECCV 2020）→ Instant-NGP → 3D Gaussian Splatting（SIGGRAPH 2023）**
2. 复现：
   - NeRF：**nerfstudio**（低门槛，自带数据集与可视化）
   - 3DGS：**graphdeco-inria/gaussian-splatting**（官方实现，跑通训练 + 渲染）
3. 理解：显式 vs 隐式表示、球谐、光度损失、从多视图恢复几何（直接衔接你的 SfM 底子！）
- ✅ **产出**：对"自己拍的多视图场景"重建出 3DGS，并渲染**新视角视频**（强烈建议，可直接作为世界模型素材）

### 阶段 4：世界模型 World Model + VLA（第 9-10 周）
**目标：理解 action-conditioned / video world model，并做出一个最小 demo。**
1. 论文主线：
   - **DreamerV3**（Hafner，world model + RL 控制）
   - **Video World Model / Cosmos（NVIDIA）**
   - **Genie / Genie 2 / Genie 3**（DeepMind，action-conditioned 可交互世界模型）
   - **V-JEPA（Meta，自监督视频世界模型）**
   - **VLA：RT-2 / Open X-Embodiment / π0（Physical Intelligence）**
2. 关键概念：**action-conditioned（以动作作条件预测下一帧/未来）**、**spatial-aware**、**scene completion**、**novel view synthesis**
3. 最小落地：复现一个 **"以相机位姿/动作作条件 → 生成下一帧"** 的小型 world model（可用你阶段 2/3 的生成模型改装）
- ✅ **产出**：一个 action-conditioned 的"下一帧预测/新视角预测"小项目 + 复现笔记

### 阶段 5：整合 + 简历化（第 11-12 周）
- 把产出整理成 2-3 个完整项目，写成简历条目（见下文）
- 读 5-8 篇 target 组近作，准备"1 分钟讲清某篇论文"和"我能贡献什么"

---

## 三、简历可写的 3 个落地项目（重点）

> 这 3 个项目分别对应岗位的三个子方向，且都建立在你的 SfM 底子上，风险低、可信度高。

### 项目 A：基于多视图采集的 3DGS 新视角生成（对应 3DGS / 新视角生成 / 场景补全）
- **做什么**：用手机/无人机围绕一个场景拍多视角照片 → 跑 **3D Gaussian Splatting** 重建 → 渲染任意新视角视频
- **产出物**：新视角渲染视频、训练收敛曲线、重建 PSNR/SSIM 指标
- **简历可写**："基于自采多视图影像完成 3D Gaussian Splatting 新视角生成与场景重建，理解显式 3D 表示与多视图几何，实现从摄影测量 SfM 到神经渲染的迁移，渲染新视角视频用于世界模型场景补全与数据增强。"

### 项目 B：Depth/Pose 条件视频生成（对应 Video Diffusion / DiT / 3D-aware）
- **做什么**：在开源视频扩散模型上，把 **深度图 / 相机位姿** 作为条件，生成可控视频（例如输入单张图 + 相机轨迹 → 输出环绕/前进视频）
- **产出物**：可控视频 demo、条件注入方式说明
- **简历可写**："在视频扩散框架中探索 Depth/Pose 条件注入，实现 3D-aware 可控视频生成，理解时空注意力与相机位姿建模。"

### 项目 C：Action-conditioned 视频世界模型（对应 World Model / VLA / action-conditioned）
- **做什么**：做一个"输入当前帧 + 动作/位姿 → 预测下一帧"的最小 world model（可基于 latent diffusion，或用 V-JEPA 思路做自监督）
- **产出物**：下一帧/轨迹预测 demo、对比 baseline 的 FVD/PSNR
- **简历可写**："复现并构建 action-conditioned 视频世界模型，以相机/动作条件预测未来帧，探索其在仿真数据生成与 VLA 训练中的应用。"

---

## 四、优先论文清单（按重要度）

**扩散/视频**
- DDPM（Denoising Diffusion Probabilistic Models, 2020）
- DDIM（2020）
- Classifier-Free Diffusion Guidance（2021）
- Latent Diffusion Models / Stable Diffusion（2022）
- Video Diffusion Models（Ho et al., 2022）
- Scalable Diffusion Models with Transformers（**DiT**, 2023）
- Stable Video Diffusion（2023）
- VideoCrafter / AnimateDiff / CogVideoX / Open-Sora / Wan（选 1-2 精读）

**3D 表示/新视角**
- NeRF（2020）
- Instant Neural Graphics Primitives（Instant-NGP, 2022）
- 3D Gaussian Splatting for Real-Time Radiance Field Rendering（2023）

**世界模型 / VLA**
- DreamerV3（2023）
- V-JEPA（Meta, 2024）
- NVIDIA Cosmos / Video World Models（2024-2025）
- Genie / Genie 2 / Genie 3（DeepMind，action-conditioned）
- RT-2 / Open X-Embodiment / π0（VLA）

---

## 五、主要开源/学习资源（请自行确认最新地址）

- **PyTorch 教程**：PyTorch 官方 tutorials（Tensors + Datasets & DataLoaders + Training a Classifier）
- **课程**：Karpathy《Neural Networks: Zero to Hero》+ Stanford CS231n（CNN/CV 经典）+ 扩散/生成课程
- **3DGS 复现**：`graphdeco-inria/gaussian-splatting`
- **NeRF 复现**：`nerfstudio-project/nerfstudio`（最友好）
- **Diffusion 复现**：`lucidrains/denoising-diffusion-pytorch`（教学向）
- **视频扩散**：`guoyww/AnimateDiff`、`hpcaitech/Open-Sora`、`THUDM/CogVideoX`、`Stability-AI/stable-video-diffusion`、`wan-video/Wan2.x`
- **世界模型**：`danijar/dreamerv3`、`facebookresearch/vjepa`、`NVIDIA/Cosmos`
- **中文社区持续跟进**：机器之心、PaperWeekly、arXiv（关注 video generation / world model / 3DGS 关键词）

---

## 六、与岗位关键词的"对表"

把学习产出和简历表述一一对应到招聘要求：
- **Video Diffusion / DiT** → 阶段 1-2 项目 B
- **Depth / Pose / Point Cloud / 3DGS 结合** → 阶段 3 + 项目 A/B
- **3D-aware / Spatial-aware** → 项目 B / C
- **Action-conditioned World Model** → 项目 C
- **仿真 / VLA 数据生成 / 场景补全 / 新视角生成** → 项目 A / C
- **PyTorch、论文阅读、代码复现** → 阶段 0-2 + 复现笔记

---

## 七、给面试的 3 句"定位话术"
1. "我的核心能力是多视图几何与三维重建（SfM/MVS、位姿估计、点云），我正在把它迁移到 3D 神经渲染（3DGS/NeRF）和世界模型方向。"
2. "我复现/跑通了 [视频扩散 / 3DGS / action-conditioned world model] 的最小 demo，理解时空注意力、相机位姿条件与 3D-aware 生成。"
3. "我能贡献：真实场景的 3D 重建与位姿估计工程能力 + 对 diffusion/DiT/world model 的快速上手，以及把 3D 信息接入视频生成的研究热情。"

---

## 八、落地节奏建议
- 若 **2 个月内投递**：优先做 **阶段 0 + 阶段 3（3DGS）**，先补齐 PyTorch 和 3DGS 实操，简历即可具备 3D Vision 竞争力；Diffusion/世界模型作为"正在学习/复现中"写进简历。
- 若 **3-4 个月后投递**：完整走完阶段 0-4，把项目 B/C 做出 demo，竞争力最强。

> ⚠️ 诚实提示：以上"正在复现/跑通"的表述需真实完成再写进简历，否则面试复盘会被追问。建议每一步都以"能重启并讲清代码"为标准。
