# vision-worldmodel-projects

image → colmap pose & points → 3DGS → camera trajectory → novel view render

scripts/trajectory.py    — generate camera paths (orbit / push / zoom)
scripts/eval_metrics.py  — PSNR / SSIM / LPIPS against GT frames
notebooks/               — Kaggle pipeline notebook

No GPU needed for trajectory generation or eval.
Full training runs on Kaggle (free T4 GPU).
