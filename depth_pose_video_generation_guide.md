# 项目 B：Depth / Pose 条件可控视频生成 —— 从零到跑通的实操脚本

> 目标：给定**一张图片**（或一段视频）+ **相机轨迹 / 深度** → 生成**可控、3D 一致**的视频（环绕 / 前进 / 缩放）。
> 对应岗位关键词：**Video Diffusion / DiT、视频编辑、可控视频生成、3D-aware、Spatial-aware、Depth / Pose / Camera 条件**。
> 前置：最好已完成项目 A（3DGS），因其产出的**相机位姿、深度、点云**正好是这里的条件输入。
> 预计：环境 0.5-1 天 + 推理生成（单次 1-3 分钟/段）。

---

## 第 0 步：先想清楚"条件"从哪来（关键设计）
项目 B 的核心是让**可控性**成立，条件有两种：
- **相机位姿（Camera Pose）**：指定一条相机轨迹（环绕 / 前推 / 缩放），让视频沿轨迹"飞行"。
- **深度（Depth）**：用单目深度估计得到场景深度，用于点云化 / 视差引导，保证几何一致。

> 推荐直接从**项目 A 的 3DGS 场景里导出真实相机轨迹**作为条件，这样 A→B 形成一条完整故事线，面试很加分。

---

## 第 1 步：选方案（参考公开方法，择一即可）

| 方案 | 做什么 | 适用 |
|---|---|---|
| **ViewCrafter** | 单图 + 相机位姿 → 新视角视频 | 最贴合需求，推荐主选 |
| **CameraCtrl** | 给已有视频附加相机运动控制 | 编辑/控制已有内容 |
| **MotionCtrl** | 相机 + 物体双运动控制 | 更自由的控制 |
| **CamCo** | 相机可控、3D 一致视频生成 | 精度导向 |

> 它们大多基于 **Stable Video Diffusion / Open-Sora** 这类 video diffusion 底座，正好覆盖岗位的 **Video Diffusion / DiT** 关键词。

---

## 第 2 步：环境搭建（以 ViewCrafter 为例）

```bash
git clone --recursive https://github.com/<ViewCrafter 仓库>.git
cd ViewCrafter
conda create -n viewcrafter python=3.10 -y
conda activate viewcrafter
pip install -r requirements.txt
# 安装对应 CUDA 版 PyTorch（与显卡匹配）
# 下载模型权重（SVD 底座 + 训练好的 checkpoint）到 checkpoints/
```

**单目深度（可选分支，用于 depth 条件）**：
```bash
pip install -U timm
# Depth Anything V2（推荐）：
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2 && pip install -r requirements.txt
# 推理：python run.py --encoder vitl --img-path <图片> --outdir <输出>
```
> 用它在输入图上生成深度图，作为 depth 条件或点云化基础。

---

## 第 3 步：准备输入
1. **一张清晰、静态、纹理丰富的场景图**（可用项目 A 数据集中的一张，或手机拍一张）。
2. **相机轨迹**：以相机外参序列（extrinsics / pose，如 Nerfstudio 或 gaussian-splatting 的 poses 格式）给出运动：
   - 环绕：沿以物体为中心的圆轨迹
   - 前进 / 缩放：相机沿视线前进或拉近
3. 若用 depth 分支：对输入图跑 Depth Anything V2 得到 `depth.npy` / 深度图。

---

## 第 4 步：生成可控视频（推理）

```bash
# 以单图 + 相机位姿生成新视角视频（ViewCrafter 式）
python infer.py \
   --image <输入图> \
   --pose_path <轨迹 poses> \
   --output <输出目录> \
   --num_frames 25 --fps 8
# 输出一段沿轨迹移动的 mp4 + 逐帧图
```

**深度引导分支（可选）**：把 `depth.npy` 作为条件注入，或先生成点云再渲染，做几何一致性校正。

---

## 第 5 步：量化评测（写进简历）

- **与 GT 对比**：如果场景来自项目 A 的 3DGS，可用其各视角渲染作为参考帧，算 **PSNR / SSIM / LPIPS**。
- **一致性 / 3D 稳定性**：用相邻帧光流误差、或对生成视频的多帧做特征匹配，看几何漂移。
- **可控性验证**：给定**两条不同相机轨迹**，生成两条视频，证明输出随条件变化（这是"可控生成"最有说服力的证据）。
- 可选：**FVD / FID** 评估生成视频分布（需要参考视频集）。

---

## 第 6 步：常见问题 & 调参

| 问题 | 解决 |
|---|---|
| 视频抖动 / 几何漂移 | 增加帧数、降低相机移动幅度、提高 cfg、用更稳的底座 |
| 前景/主体变化 | 用更强的 identity 条件（参考图）、减少随意运动 |
| 显存不足 | 降分辨率 / 减帧数 / 用更小模型权重 |
| 生成物体"变形" | 轨迹太激进；先做慢速小角度验证 |
| 只想要"镜头运动、场景不变" | 用 CameraCtrl 类方案，锁住内容只控相机 |

---

## 第 7 步：写成简历条目（对应 Video Diffusion / DiT / 3D-aware）

> 参考文案（**真实跑通后写**）：
>
> **Depth / Pose 条件可控视频生成    2026.xx**
> - 基于 [Video Diffusion / DiT] 底座（如 ViewCrafter / CameraCtrl），实现**相机位姿驱动的可控视频生成**，给定单图 + 相机轨迹生成环绕 / 前进 / 缩放视频；
> - 引入单目深度估计（Depth Anything V2）生成场景深度，作为 depth 条件与几何一致性约束，提升 3D-aware 生成质量；
> - 通过**控制相机轨迹**验证可控性（不同轨迹 → 不同运动），并在参考视角上以 PSNR / SSIM / LPIPS 定量评估；
> - 探索将 3DGS / SfM 重建产出的**位姿、深度、点云**作为条件接入视频生成，服务于世界模型与视频编辑。

**面试一句话**：
> "我基于 video diffusion 实现了相机位姿 + 深度可控的视频生成，能沿指定轨迹渲染出 3D 一致的视频，并定量验证了可控性。"

---

## 附：与岗位的衔接
- 这直接对应"**研究 Video Diffusion / DiT、视频编辑、可控视频生成**"与"**探索视频生成与 Depth、Pose、Point Cloud、3DGS 等三维信息结合**"。
- 产出可控视频，可作为**项目 C（世界模型）**的训练数据 / 测试基准。
