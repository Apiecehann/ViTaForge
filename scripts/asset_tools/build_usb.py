"""生成简化版 USB peg 网格(机身长方体 + 矩形 USB-A 插口, 插口朝 -Z)。
用法: python scripts/asset_tools/build_usb.py [out.obj]   (默认 /tmp/usb_assets/USB.obj)
env: ~/miniconda3/envs/UniVTAC/bin/python
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/usb_assets/USB.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

BODY = [19.9, 9.5, 52.0]      # 机身 宽×厚×长 (mm)
CONN = [12.0, 4.5, 14.0]      # 插口 标准 USB-A 12×4.5, 长14(~3mm 嵌入机身)
body = trimesh.creation.box(extents=BODY); body.apply_translation([0, 0, BODY[2] / 2])
conn = trimesh.creation.box(extents=CONN); conn.apply_translation([0, 0, -CONN[2] / 2 + 3.0])
usb = trimesh.boolean.union([body, conn])         # 水密单体
usb.apply_translation(-usb.bounds.mean(axis=0))   # 居中到原点 (插口端在 -Z)
usb.apply_scale(0.001)                            # mm → m
usb.export(out)
print(f"saved {out}  extents(mm)={np.round(usb.extents*1000,2)}  watertight={usb.is_watertight}")
