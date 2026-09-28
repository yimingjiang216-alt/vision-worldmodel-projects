# -*- coding: utf-8 -*-
"""把训练好的 Video DiT 导出为 ONNX, 供浏览器端 onnxruntime-web 实时推理.

导出单步去噪算子 f(x_t, t, actions) -> x0_pred, DDIM 循环在 JS 端实现,
这样"可视世界模型"完全在浏览器里跑: 用户拨动作 -> 实时看到生成的未来视频.
"""
import argparse
import numpy as np
import torch
import data as D
from model import VideoDiT
from diffusion import Diffusion


def export(ckpt, out, n_steps_ddim=25):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    T_, sz = a["T"], a["size"]
    model = VideoDiT(img_size=sz, patch=8, in_ch=3, T=T_, action_dim=2,
                     dim=a["dim"], depth=a["depth"], heads=4, cond_dim=a["dim"])
    model.load_state_dict(ck["model"]); model.eval()

    # 固定 batch=1 的输入形状
    x = torch.randn(1, T_, 3, sz, sz)
    t = torch.zeros(1, dtype=torch.long)
    acts = torch.zeros(1, T_, 2)
    torch.onnx.export(
        model, (x, t, acts), out,
        input_names=["x_t", "t", "actions"],
        output_names=["x0_pred"],
        dynamic_axes=None,
        opset_version=18,
    )
    print("exported ->", out)

    # 校验: ONNX Runtime 推理与 PyTorch 一致
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    xt = np.random.RandomState(0).randn(1, T_, 3, sz, sz).astype(np.float32)
    tt = np.array([137], dtype=np.int64)
    at = np.random.RandomState(1).randn(1, T_, 2).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(xt), torch.from_numpy(tt),
                    torch.from_numpy(at)).numpy()
    got = sess.run(None, {"x_t": xt, "t": tt, "actions": at})[0]
    err = float(np.abs(ref - got).max())
    print(f"onnx vs torch max|diff| = {err:.2e}  {'OK' if err < 1e-4 else 'MISMATCH'}")
    return out, dict(T=T_, size=sz, dim=a["dim"], depth=a["depth"],
                     n_steps=a["n_steps"], target=a.get("target", "eps"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt8/video_dit.pt")
    ap.add_argument("--out", default="web/video_dit.onnx")
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    _, meta = export(args.ckpt, args.out)
    import json
    json.dump(meta, open(args.out.replace(".onnx", "_meta.json"), "w"),
              ensure_ascii=False, indent=2)
    print("meta:", json.dumps(meta))
