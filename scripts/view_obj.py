"""Interactive 3D preview of a mesh (OBJ/STL/PLY/GLB...) in a pyglet window.

Usage:
    python scripts/view_obj.py [path/to/mesh.obj]

Controls (trimesh viewer):
    drag        rotate
    scroll      zoom
    right-drag  pan
    a           toggle XYZ axis
    w           toggle wireframe
    g           toggle grid
    q / Esc     quit
"""

import sys
import numpy as np
import trimesh

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/usb_simple.obj"

m = trimesh.load(path, process=False)
if isinstance(m, trimesh.Scene):
    m = m.to_geometry()
m = trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.faces), process=False)
m.visual = trimesh.visual.ColorVisuals(m, face_colors=[150, 160, 180, 255])

ext = m.extents * 1000.0
print(f"[view_obj] {path}")
print(f"[view_obj] {len(m.vertices)} verts / {len(m.faces)} faces, extents {np.round(ext,2)} mm, "
      f"watertight={m.is_watertight}")

# XYZ axis sized to the object so orientation is obvious
axis = trimesh.creation.axis(origin_size=m.scale * 0.01, axis_length=m.scale * 0.6)

scene = trimesh.Scene([m, axis])
scene.show(caption=f"{path}  ({len(m.faces)} faces)", smooth=False)
