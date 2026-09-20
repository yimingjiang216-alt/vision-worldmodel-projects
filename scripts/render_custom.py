# Custom orbit renderer for a trained 3DGS model.
#
# Run INSIDE the gaussian-splatting repo directory (so `from gaussian_renderer
# import render` and `from scene.gaussian_model import GaussianModel` resolve).
#
#   python render_custom.py
#     --model_path /kaggle/working/output/scene_run1
#     --output_dir /kaggle/working/output/orbit
#     --frames 120
#
# It reads the training cameras.json (written next to the trained model) only to
# recover image resolution + intrinsics + a rough scene center/scale. The orbit
# itself is procedural: N cameras circling the recovered center, looking at it.

import argparse
import json
import math
import os

import numpy as np
import torch
import torchvision

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel


def focal_to_fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def look_at_view_matrix(eye, center, up):
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = eye - center
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    R = np.stack([x, y, z], axis=1)  # world->cam rotation (cols)
    t = -R.T @ eye
    Rt = np.eye(4)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    return torch.tensor(Rt, dtype=torch.float32, device="cuda")


class MiniCam:
    def __init__(self, width, height, fovy, fovx, world_view_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = 0.01
        self.zfar = 100.0
        self.world_view_transform = world_view_transform
        self.projection_matrix = get_projection_matrix(self.znear, self.zfar, fovx, fovy).transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def get_projection_matrix(znear, zfar, fov_x, fov_y):
    tan_half_y = math.tan(fov_y / 2)
    tan_half_x = math.tan(fov_x / 2)
    top = tan_half_y * znear
    bottom = -top
    right = tan_half_x * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--radius_scale", type=float, default=1.0,
                    help="orbit radius as a multiple of the recovered scene radius")
    ap.add_argument("--height_scale", type=float, default=0.3,
                    help="camera height as a multiple of the recovered scene radius")
    args = ap.parse_args()

    model_path = args.model_path
    ply = os.path.join(model_path, "point_cloud", "iteration_7000", "point_cloud.ply")
    if not os.path.exists(ply):
        # fall back to whatever iteration exists
        pc_root = os.path.join(model_path, "point_cloud")
        its = [d for d in sorted(os.listdir(pc_root)) if d.startswith("iteration_")]
        if not its:
            raise SystemExit(f"no trained point cloud under {pc_root}")
        ply = os.path.join(pc_root, its[-1], "point_cloud.ply")
        print(f"using {ply}")

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply)

    # recover intrinsics + scene center from training cameras.json
    cams_json = os.path.join(model_path, "cameras.json")
    if not os.path.exists(cams_json):
        raise SystemExit(f"no {cams_json}; run training first")
    with open(cams_json) as f:
        cams = json.load(f)

    first = cams[0]
    w, h = int(first["width"]), int(first["height"])
    fx, fy = first["fx"], first["fy"]
    fovx = focal_to_fov(fx, w)
    fovy = focal_to_fov(fy, h)

    positions = np.array([c["position"] for c in cams])
    center = positions.mean(axis=0)
    radius = float(np.linalg.norm(positions - center, axis=1).max())
    if radius <= 0:
        radius = 1.0
    print(f"scene center={center}, radius={radius}, res={w}x{h}")

    R = radius * args.radius_scale
    H = radius * args.height_scale

    os.makedirs(args.output_dir, exist_ok=True)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # a PipelineParams-like minimal object; defaults match train.py
    class Pipe:
        debug = False
        antialiasing = False
        convert_SHs_python = False
        compute_cov3D_python = False
    pipe = Pipe()

    n = args.frames
    with torch.no_grad():
        for i in range(n):
            a = 2 * math.pi * i / n
            eye = np.array([center[0] + R * math.cos(a),
                            center[1] + R * math.sin(a),
                            center[2] + H])
            view = look_at_view_matrix(eye, center, [0, 0, 1])
            cam = MiniCam(w, h, fovy, fovx, view)
            out = render(cam, gaussians, pipe, bg)
            img = out["render"].clamp(0, 1)
            torchvision.utils.save_image(img, os.path.join(args.output_dir, f"{i:05d}.png"))
            if (i + 1) % 20 == 0:
                print(f"rendered {i + 1}/{n}")

    print(f"done -> {args.output_dir}")


if __name__ == "__main__":
    main()
