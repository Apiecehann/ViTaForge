"""Open a USD asset in the Isaac Sim GUI for quick visual inspection.

Usage:
    python scripts/view_usd.py [path/to/asset.usd]

Spawns the asset (geometry only, no physics) onto a ground plane with a dome
light and frames the camera on it. Keeps the viewport open until you close it.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View a USD asset in Isaac Sim.")
parser.add_argument("usd", nargs="?", default="assets/objects/USB.usd",
                    help="Path to the USD file to view.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False  # GUI

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

usd_path = os.path.abspath(args_cli.usd)
print(f"[view_usd] opening: {usd_path}")

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cpu"))

# ground + light
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)).func(
    "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
)

# spawn the asset, lifted so it rests above the ground
cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
cfg.func("/World/Asset", cfg, translation=(0.0, 0.0, 0.05))

# frame a small (~cm-scale) object
sim.set_camera_view(eye=(0.16, 0.16, 0.11), target=(0.0, 0.0, 0.03))

sim.reset()
print("[view_usd] viewport ready — close the window to exit.")
while simulation_app.is_running():
    sim.step()

simulation_app.close()
