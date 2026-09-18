# Vision / World Model Projects

作者：蒋一鸣（Yiming Jiang）｜ 遥感科学与技术 本科生 ｜ 方向：三维视觉 / 三维重建 / SLAM / 世界模型 / 视频生成

本仓库用于记录与展示我在 **三维视觉、三维建模、SLAM 视觉定位、世界模型与视频生成** 方向的学习与项目进展。
所有条目均标注真实状态：**✅ 已完成** 或 **🟡 进行中**，不夸大、不虚构。

## 项目总览

### ✅ 已完成（真实项目）
- **SfM / MVS 大规模场景三维重建**（无人机倾斜摄影）：特征提取与匹配 → BA 光束法平差 → 稠密点云（MVS）→ 网格重建 → 纹理映射；控制点总误差 < 0.1m，满足 1:2000 测图规范。
- **摄影测量与计算机视觉核心算法实现**：独立实现 SIFT、Harris、Forstner、Canny、LoG 等特征提取与匹配算法，以及单像空间后方交会（相机位姿估计）；基于 SAM 完成影像分割与目标提取。
- **无人机航测空中三角测量与精度优化**：相机标定与内参优化、同名点自动匹配、光束法区域网平差（PATB）、坐标系统转换。
- **多光谱遥感影像解译与多指数分层分类**：决策树 + 纹理 + 多光谱指数分层分类（12 类），混淆矩阵精度评定（85.53%）。
- **卫星遥感数据几何重定位**：NetCDF 大规模数据读取、GLT 查找表、几何重定位与坐标转换。
- **ArcGIS 空间分析与自动化**：空间分析流程构建、插值、叠加分析与制图，C++ 自动化。

### 🟡 进行中 / 学习路线
- **3D Gaussian Splatting 场景重建与新视角生成**（详见 `3dgs_novel_view_guide.md`）
- **Depth / Pose 条件可控视频生成**（详见 `depth_pose_video_generation_guide.md`）
- **Action-conditioned 视频世界模型**（详见 `action_world_model_guide.md`）
- **世界模型 / 视频生成 学习路线**（详见 `learning_roadmap.md`）

## 文件说明
| 文件 | 说明 |
|---|---|
| `completed_photogrammetry_cv_projects.md` | 已完成项目的详细说明（真实成果） |
| `3dgs_novel_view_guide.md` | 3DGS 新视角生成 从零实操指南（进行中） |
| `depth_pose_video_generation_guide.md` | Depth/Pose 条件可控视频生成 实操指南（进行中） |
| `action_world_model_guide.md` | Action-conditioned 世界模型 实操指南（进行中） |
| `learning_roadmap.md` | 世界模型 / 视频生成 学习路线 |

> 说明：本项目以"多视图几何 + 三维重建"为核心能力，覆盖三维建模、SLAM 视觉定位、世界模型与视频生成方向。
