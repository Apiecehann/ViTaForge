"""生成一个薄薯片/细条网格(轻微弧度, 薄片状)。
用法: python scripts/asset_tools/build_chip.py [out.obj]   (默认 /tmp/chip_assets/CHIP.obj)
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/chip_assets/CHIP.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

L, W, T = 82.5, 18.0, 2.4       # 长 × 宽 × 厚 (mm) —— 放大 1.5 倍(chip7 那版)
ARC = 13.5                      # 纵向弧度: 两端比中间抬高 ~13.5mm

box = trimesh.creation.box(extents=[L, W, T])
box = box.subdivide_to_size(1.8)            # 细分, 弧度才平滑、tet 才合理
V = box.vertices.copy()
V[:, 2] += ARC * (V[:, 0] / (L / 2)) ** 2   # 沿长度方向轻微上弯
chip = trimesh.Trimesh(V, box.faces, process=True)
chip.merge_vertices()

chip.apply_translation(-chip.bounds.mean(axis=0))   # 居中
chip.apply_scale(0.001)                             # mm → m
chip.export(out)
print(f"saved {out}  extents(mm)={np.round(chip.extents*1000,2)}  "
      f"verts={len(chip.vertices)} faces={len(chip.faces)} watertight={chip.is_watertight}")
