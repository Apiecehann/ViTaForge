"""把 bake_tet.py 产出的 npz 写进 surface USD 的 /<name>/mesh 上(tet_* 四个属性)。
写完加载器就会跳过 fTetWild、直接用这个 tet(秒加载)。
用法(pxr 不在标准路径, 需用 Isaac 自带的 extscache):
  EXT=~/miniconda3/envs/UniVTAC/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-*
  PYTHONPATH=$EXT LD_LIBRARY_PATH=$EXT/bin:$EXT/pxr/.libs \\
    python scripts/asset_tools/write_tet_to_usd.py <asset.usd> <tet.npz>
"""
import sys, numpy as np
from pxr import Usd, Sdf, Vt

usd_path, npz_path = sys.argv[1], sys.argv[2]
d = np.load(npz_path)
s = Usd.Stage.Open(usd_path)
mesh = [p for p in s.Traverse() if p.GetTypeName() == 'Mesh'][0]


def set_vec3f(n, a): mesh.CreateAttribute(n, Sdf.ValueTypeNames.Float3Array).Set(Vt.Vec3fArray.FromNumpy(a.reshape(-1, 3)))
def set_uint(n, a): mesh.CreateAttribute(n, Sdf.ValueTypeNames.UIntArray).Set(Vt.UIntArray.FromNumpy(a.reshape(-1)))


set_vec3f("tet_points", d["tet_points"].astype(np.float32))
set_uint("tet_indices", d["tet_indices"].astype(np.uint32))
set_vec3f("tet_surf_points", d["surf_points"].astype(np.float32))
set_uint("tet_surf_indices", d["surf_indices"].astype(np.uint32))
s.Save()
print(f"{usd_path} <- {len(d['tet_indices'])//4} tets ({mesh.GetPath()})")
