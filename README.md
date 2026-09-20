# vision-worldmodel-projects

image -> colmap pose & points -> 3DGS -> orbit render -> novel view video

scripts/trajectory.py     - camera paths (orbit / push / zoom)
scripts/render_custom.py  - orbit render of a trained 3DGS model (run inside gaussian-splatting/)
scripts/eval_metrics.py   - PSNR / SSIM / LPIPS against GT frames
notebooks/                - Kaggle pipeline (COLMAP -> train -> render -> video)

Local: no GPU needed for trajectory or eval.
Training: Kaggle free T4 GPU (run notebooks/kaggle_pipeline.ipynb).
