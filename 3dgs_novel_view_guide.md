# 项目 A：3D Gaussian Splatting 新视角生成 —— 从零到跑通的实操脚本

> 目标：用你自己拍摄/采集的多视图影像 → COLMAP 恢复相机位姿 → 训练 3DGS → 渲染任意新视角视频。
> 对应岗位关键词：**3DGS、新视角生成（Novel View Synthesis）、场景重建、多视图几何、场景补全**。
> 预计耗时：环境搭建 0.5-1 天 + 数据采集半天 + 训练（单场景 10 分钟-1 小时，视 GPU 而定）。

---

## 第 0 步：先确认产出长什么样
- 一个可复现的 3DGS 项目文件夹（含数据集、训练脚本、训练日志）
- 一段**新视角环绕 / 前进视频**（这可是世界模型/视频生成方向的绝佳素材）
- 一张 PSNR / SSIM 评测表 + 训练 visual 对比图
- 一段能写进简历的项目描述（见第 9 步）

---

## 第 1 步：硬件 / 环境检查
**推荐：Linux + NVIDIA GPU（至少 8-12GB 显存）。**
```
# 检查显卡
nvidia-smi

# 若无 Linux，强烈建议 Windows 上用 WSL2（Ubuntu 22.04），CUDA 扩展编译更省事
```
> ⚠️ Windows 原生训练较麻烦：`diff-gaussian-rasterization`、`simple-knn` 是 CUDA 扩展，需要 **CUDA Toolkit + MSVC 编译工具链**。用 Linux 或 WSL2 几乎零配置。

**必备**：Anaconda/Miniconda、Git、CUDA（11.8 或 12.x）、Python 3.8-3.11（3DGS 建议 3.8/3.9）。

---

## 第 2 步：数据采集（最重要！决定重建质量）

### 用手机拍（最简单）
- 围绕一个**静态、纹理丰富**的场景，以**环绕 + 上下多角度**走动拍摄
- 相邻两帧**重叠率 > 70%**（越密越好），**不要大幅跳跃**
- **避免**：反光/镜面、纯白/无纹理墙面、移动物体（人、车、树叶）、过度曝光
- 建议 **100-300 张**；拍完把照片按时间顺序放一个文件夹（如 `images/`）
- 可选：用无人机倾斜摄影（你更熟练），同样能得到高质量多视图

### 目录结构（COLMAP 格式，gaussian-splatting 所需）
```
project_a/
└─ data/
   └─ scene/                # 数据集根目录
      ├─ images/           # 所有输入照片（jpg/png）
      └─ sparse/0/         # COLMAP 输出（下面自动生成）
```

---

## 第 3 步：环境搭建（Linux / WSL2 版）

```bash
# 1. 克隆官方仓库（含子模块）
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting

# 2. 用官方 environment.yml 创建 conda 环境
conda env create --python=3.9 gaussian_env        # 或直接用官方 env
conda activate <你的环境名>

# 3. 安装依赖（官方 README 方式）
pip install plyfile tqdm
# 若仓库自带 environment.yml：
# conda env create --file environment.yml && conda activate gaussian_splatting

# 4. 编译 CUDA 扩展（子模块）
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
> 报 `nvcc not found`：确认已装与 PyTorch 匹配的 **CUDA Toolkit**，并把 `/usr/local/cuda/bin` 加入 PATH。

**可选工具（强烈推荐）**：装 **nerfstudio**，用它做 COLMAP 预处理和可视化更省事。
```
pip install nerfstudio
```

---

## 第 4 步：用 COLMAP 恢复相机位姿（SfM 预处理）

### 方式一：直接跑 COLMAP 命令（最贴近你 SfM 底子）
```bash
cd data/scene
# 1. 特征提取
colmap feature_extractor \
   --database_path database.db \
   --image_path images

# 2. 特征匹配（小数据用 exhaustive，大数据用 sequential/voce）
colmap exhaustive_matcher \
   --database_path database.db

# 3. 增量式 SfM / 建图 → 生成 sparse/0
colmap mapper \
   --database_path database.db \
   --image_path images \
   --output_path sparse
# 结果会得到 sparse/0（含 cameras.bin / images.bin / points3D.bin）

# 4.（可选）畸变校正生成 undistor 后图像
colmap image_undistorter \
   --image_path images \
   --input_path sparse/0 \
   --output_path undistorted \
   --output_type COLMAP
```

### 方式二：用 nerfstudio 一条命令（最简单）
```bash
ns-process-data images \
   --data <你的照片文件夹> \
   --output-dir data/scene \
   --sfm colmap
# 会生成 gaussian-splatting 能直接读的 transforms.json + sparse 结构
```
> 若用 nerfstudio 输出，某些 3DGS 教程会读 `transforms.json`；为保险，仍建议保留 `sparse/0`。

---

## 第 5 步：训练 3DGS

```bash
# 回到仓库根目录
python train.py \
   -s <路径>/data/scene \
   -m <路径>/output/scene_run1 \
   --iterations 30000 \
   --save_iterations 7000 30000 \
   --test_iterations 7000 30000 \
   --resolution 1
```
- `-s` = 数据集位置（含 `images/` 和 `sparse/0/`）
- `-m` = 输出模型/日志目录
- `--iterations`：默认 30000，**快速试探可先用 7000**
- `--resolution 1`：若显存不足降到 `0.5`，或数据过大用 `-r`
- 训练完会在 `-m` 下生成 `point_cloud/iteration_30000/`（含 `point_cloud.ply`）和 `cameras.json`

**看训练可视化**：`-m` 目录下有 `.../output/...`，可用 tensorboard/vis 脚本观察 PSNR 收敛（仓库提供 `--viewer` 可实时预览，需装 gsplat viewer）。

---

## 第 6 步：渲染新视角 + 导出视频

```bash
# 1. 在训练视角上渲染重建图像 + 计算指标
python render.py -m <路径>/output/scene_run1

# 2. 生成你的"新视角环绕视频"（重点交付物）
#    用仓库的 make_video 脚本：
python scripts/make_video.py \
   -m <路径>/output/scene_run1 \
   -o <路径>/output/scene_run1/video \
   --train_views 0 --test_views 0 --trajectory circular
# 或手动写一个脚本，从 poses 生成一条相机轨迹，逐帧渲染后 ffmpeg 合成
```
> 没现成脚本时，可用 `render.py` 的 pose 参数化思路：在测试/自定义相机位姿上逐帧渲染，再：
> ```
> ffmpeg -framerate 30 -i frame_%04d.png -pix_fmt yuv420p novel_view.mp4
> ```

---

## 第 7 步：量化评测（写进简历的指标）

```bash
python metrics.py -m <路径>/output/scene_run1
# 输出 PSNR / SSIM / LPIPS（需 pip install lpips）
```
- 记录并对比在不同 `--iterations`（如 7000 / 30000）下的 **PSNR、SSIM、LPIPS**
- 可选：与 NeRF（nerfstudio 的 nerfacto）同数据集对比，写进"方法对比"更有说服力
- 可做一个 2 张图对比（原图 vs 重建渲染图）放进简历/作品集

---

## 第 8 步：常见问题 & 调参

| 问题 | 解决 |
|---|---|
| 重建后场景"糊" / 有 hole | 提高照片重叠率、增加帧数、`--resolution 1`、增加迭代到 30000 |
| 显存爆 `CUDA out of memory` | 降 `--resolution 0.5`、减少迭代、用更小场景 |
| COLMAP 匹配失败/场景失败 | 照片太少或重叠不足；用 `--sfm colmap`，检查是否含 EXIF |
| 编译 CUDA 扩展报错 | 确认 CUDA Toolkit 版本与 PyTorch 匹配、装 `build-essential`、MSVC（Windows） |
| 出现大量 floating 高斯/噪点 | 用官方 `--densify_*` 参数或后处理 `point_cloud.ply` 统计 + 视锥剔除 |
| 背景/天空重建差 | 用 nerfstudio 的 mask / background 设置，或裁剪掉天空区域 |

**提质量小技巧**：优先保证**数据集质量**（重叠、纹理、无运动物体），Gaussian Splatting 对数据质量远比迭代次数敏感。

---

## 第 9 步：写成简历条目（对应 3DGS / 新视角生成）

> 参考文案（**请在你真实跑通后再写**）：
>
> **3D Gaussian Splatting 多视场景重建与新视角生成    2026.xx**
> - 基于自采多视图影像（约 N 张，环绕+多角度），使用 COLMAP 完成特征提取、匹配与增量式 SfM 重建，恢复相机内参与位姿；
> - 在 NVIDIA 3D Gaussian Splatting 框架上完成场景重建与显式 3D 表示训练，渲染任意新视角视频，实现新视角生成（Novel View Synthesis）与场景补全；
> - 在测试视角上以 PSNR / SSIM / LPIPS 定量评估（如 PSNR ~xx dB / SSIM ~0.xx），与 NeRF 基准对比；
> - 理解从摄影测量 SfM/MVS 到神经渲染的迁移，具备多视图几何、相机位姿估计与显式 3D 表示的工程落地能力。

**面试一句话**：
> "我把 SfM/MVS 的多视图几何能力迁移到了 3D Gaussian Splatting，用自己采集的场景做完了新视角重建与渲染，并量化对比了 PSNR/SSIM/LPIPS。"

---

## 附：本项目对"世界模型/视频生成"的衔接价值
- 3DGS 渲染出的**新视角视频 / 环绕轨迹**，可直接作为**世界模型 / 视频扩散模型的训练数据或条件**（深度、位姿、点云都在重建中天然产生）。
- 这正好对应招聘里"**探索视频生成与 Depth / Pose / Point Cloud / 3DGS 等三维信息结合**"，是你后续做项目 B/C 的桥。
