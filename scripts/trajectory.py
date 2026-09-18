"""
Camera trajectory generator for 3DGS novel view rendering.

Generates a sequence of camera poses along a user-defined path
(orbit / push / zoom / custom spline) and exports them in a format
readable by the 3D Gaussian Splatting renderer.

No GPU required. Run locally before uploading to Kaggle.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def look_at(eye, center, up):
    """Build a 4x4 world-to-camera (view) matrix from eye/center/up."""
    z = np.array(eye) - np.array(center)
    z = z / np.linalg.norm(z)
    x = np.cross(np.array(up), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    R = np.stack([x, y, z], axis=1)
    t = -R.T @ eye
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R.T
    M[:3, 3] = t
    return M


def look_at_inv(view_matrix):
    """Convert a 4x4 view matrix to a camera-to-world (pose) matrix."""
    R = view_matrix[:3, :3].T
    t = -R @ view_matrix[:3, 3]
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def orbit_trajectory(center, radius, height, num_frames, angle_range=360.0):
    """Circular orbit around a center point."""
    poses = []
    for i in range(num_frames):
        angle = math.radians(i / num_frames * angle_range)
        eye = np.array([
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            height,
        ])
        view = look_at(eye, center, [0, 0, 1])
        poses.append(look_at_inv(view))
    return poses


def push_trajectory(start_eye, direction, steps, step_size):
    """Move straight along a direction vector."""
    d = np.array(direction) / np.linalg.norm(direction)
    poses = []
    for i in range(steps):
        eye = np.array(start_eye) + d * step_size * i
        center = eye + d * 10  # look ahead
        view = look_at(eye, center, [0, 0, 1])
        poses.append(look_at_inv(view))
    return poses


def zoom_trajectory(center, start_radius, end_radius, height, num_frames):
    """Zoom in from start_radius to end_radius at a fixed angle."""
    poses = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        radius = start_radius + (end_radius - start_radius) * t
        eye = np.array([center[0] + radius, center[1], height])
        view = look_at(eye, center, [0, 0, 1])
        poses.append(look_at_inv(view))
    return poses


def poses_to_json(poses, output_path):
    """Export poses as a JSON file with the same convention as 3DGS cameras.json."""
    out = []
    for pid, pose in enumerate(poses):
        out.append({
            "id": pid,
            "img_name": f"traj_{pid:04d}",
            "width": 0,   # will be filled by renderer
            "height": 0,
            "position": pose[:3, 3].tolist(),
            "rotation": pose[:3, :3].tolist(),
            "fy": 0,
            "fx": 0,
        })
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(out)} poses to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate camera trajectory for 3DGS")
    parser.add_argument("--mode", default="orbit",
                        choices=["orbit", "push", "zoom"])
    parser.add_argument("--output", default="trajectory.json")
    parser.add_argument("--frames", type=int, default=60)

    # orbit params
    parser.add_argument("--center", nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--height", type=float, default=3.0)
    parser.add_argument("--angle", type=float, default=360.0)

    # push params
    parser.add_argument("--start", nargs=3, type=float, default=[0, -5, 3])
    parser.add_argument("--direction", nargs=3, type=float, default=[0, 1, 0])
    parser.add_argument("--step", type=float, default=0.5)

    # zoom params
    parser.add_argument("--start_r", type=float, default=10.0)
    parser.add_argument("--end_r", type=float, default=2.0)

    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    modes = {
        "orbit": lambda: orbit_trajectory(args.center, args.radius,
                                          args.height, args.frames, args.angle),
        "push": lambda: push_trajectory(args.start, args.direction,
                                        args.frames, args.step),
        "zoom": lambda: zoom_trajectory(args.center, args.start_r,
                                        args.end_r, args.height, args.frames),
    }
    poses = modes[args.mode]()

    # Export as 4x4 numpy flat format (compatible with 3DGS render.py custom camera)
    np.savez_compressed(args.output.replace(".json", "_poses.npz"), poses=np.array(poses))
    print(f"Exported poses.npz with shape {np.array(poses).shape}")

    # Also export a readable JSON
    poses_to_json(poses, args.output)


if __name__ == "__main__":
    main()
