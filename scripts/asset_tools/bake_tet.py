"""给一个干净网格离线烤四面体网格(fTetWild), 存成 npz。
用法: python scripts/asset_tools/bake_tet.py <in.obj> <out.npz> [edge_length_r] [epsilon]
  edge_length_r 控制 tet 数量(=仿真开销, 越小越密)；epsilon 控制表面贴合/光滑(越小越紧/越平)。
注意: 每个资产单独一个进程跑(同进程多次 tetrahedralize 会 double-free 崩)。
"""
import sys, numpy as np, trimesh, wildmeshing as wm

obj, npz = sys.argv[1], sys.argv[2]
elr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0263   # ~1.5mm 边长档(对 ~67mm 物体)
eps = float(sys.argv[4]) if len(sys.argv) > 4 else 0.001    # 紧包络 -> 表面光滑

m = trimesh.load(obj, process=False)
if isinstance(m, trimesh.Scene):
    m = m.to_geometry()
pts = np.asarray(m.vertices, float)
tri = np.asarray(m.faces, np.uint32).reshape(-1, 3)

t = wm.Tetrahedralizer(stop_quality=10, max_its=40, epsilon=eps,
                       edge_length_r=elr, skip_simplify=False, coarsen=True)
t.set_log_level(6)
t.set_mesh(pts, tri)
t.tetrahedralize()
tp, ti, _ = t.get_tet_mesh(correct_surface_orientation=True, manifold_surface=True,
                           use_input_for_wn=False)
sp, si = t.get_tracked_surfaces(); sp = sp[0]

np.savez(npz,
         tet_points=np.asarray(tp, np.float32),
         tet_indices=np.asarray(ti).reshape(-1).astype(np.uint32),
         surf_points=np.asarray(sp, np.float32),
         surf_indices=np.asarray(si).reshape(-1).astype(np.uint32))
print(f"{obj} -> {npz}: {len(tp)} pts / {len(np.asarray(ti).reshape(-1))//4} tets")
