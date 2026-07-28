"""Run the TacEx bunny-cloth UIPC demo headlessly and save an RGB video."""

import argparse
import pathlib
import subprocess

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Render the TacEx libuipc bunny-cloth demo to video.")
parser.add_argument("--output", type=str, default="outputs/bunny_cloth.mp4", help="Output mp4 path.")
parser.add_argument("--steps", type=int, default=180, help="Number of UIPC simulation steps to render.")
parser.add_argument("--fps", type=int, default=20, help="Output video framerate.")
parser.add_argument("--width", type=int, default=960, help="Camera image width.")
parser.add_argument("--height", type=int, default=540, help="Camera image height.")
parser.add_argument("--warmup", type=int, default=5, help="Warmup render steps before recording.")
parser.add_argument("--camera-eye", type=float, nargs=3, default=(3.0, 2.0, 2.0), help="Camera eye position.")
parser.add_argument("--camera-target", type=float, nargs=3, default=(0.6, 0.35, 0.15), help="Camera target position.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import omni.usd
from pxr import UsdGeom
from uipc import Transform, Vector3, builtin, view
from uipc.constitution import AffineBodyConstitution, DiscreteShellBending, ElasticModuli, NeoHookeanShell
from uipc.geometry import SimplicialComplexIO, flip_inward_triangles, label_surface, label_triangle_orient
from uipc.unit import MPa, kPa

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.timer import Timer

from tacex_uipc.sim import UipcSim, UipcSimCfg


def setup_base_scene(sim: sim_utils.SimulationContext):
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func(
        prim_path="/World/defaultGroundPlane",
        cfg=cfg_ground,
        translation=[0, -1, 0],
        orientation=[0.7071068, -0.7071068, 0, 0],
    )

    cfg_light_dome = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light_dome.func("/World/lightDome", cfg_light_dome, translation=(1, 10, 0))


def setup_libuipc_scene(scene):
    sample_dir = pathlib.Path("third_party/TacEx/source/tacex_uipc/examples/libuipc-samples").resolve()
    trimesh_path = str(sample_dir / "trimesh")
    tetmesh_path = str(sample_dir / "tet_meshes")

    cloth = scene.objects().create("cloth")
    transform = Transform.Identity()
    transform.scale(2.0)
    io = SimplicialComplexIO(transform)
    cloth_mesh = io.read(f"{trimesh_path}/grid20x20.obj")
    label_surface(cloth_mesh)
    shell = NeoHookeanShell()
    bending = DiscreteShellBending()
    moduli = ElasticModuli.youngs_poisson(10 * kPa, 0.499)
    shell.apply_to(cloth_mesh, moduli=moduli, mass_density=200, thickness=0.001)
    bending.apply_to(cloth_mesh, E=10.0)
    view(cloth_mesh.positions())[:] += 1.0
    cloth.geometries().create(cloth_mesh)

    bunny = scene.objects().create("bunny")
    transform = Transform.Identity()
    transform.translate(Vector3.UnitX() + Vector3.UnitZ())
    io = SimplicialComplexIO(transform)
    bunny_mesh = io.read(f"{tetmesh_path}/bunny0.msh")
    label_surface(bunny_mesh)
    label_triangle_orient(bunny_mesh)
    bunny_mesh = flip_inward_triangles(bunny_mesh)
    abd = AffineBodyConstitution()
    abd.apply_to(bunny_mesh, 100 * MPa)
    is_fixed = bunny_mesh.instances().find(builtin.is_fixed)
    view(is_fixed)[:] = 1
    bunny.geometries().create(bunny_mesh)


def setup_camera(sim: sim_utils.SimulationContext) -> Camera:
    camera_cfg = CameraCfg(
        prim_path="/World/VideoCamera",
        update_period=0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=3.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
    )
    camera = Camera(cfg=camera_cfg)
    return camera


def start_video_writer(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{args_cli.width}x{args_cli.height}",
            "-framerate",
            str(args_cli.fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def camera_rgb(camera: Camera) -> np.ndarray:
    frame = camera.data.output["rgb"][0, ..., :3]
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, gravity=[0.0, -9.8, 0.0])
    sim = sim_utils.SimulationContext(sim_cfg)
    setup_base_scene(sim)

    uipc_cfg = UipcSimCfg(
        dt=0.01,
        gravity=[0.0, -9.8, 0.0],
        ground_normal=[0, 1, 0],
        ground_height=-1.0,
        contact=UipcSimCfg.Contact(default_friction_ratio=0.5, default_contact_resistance=1.0, d_hat=0.01),
    )
    uipc_sim = UipcSim(uipc_cfg)
    setup_libuipc_scene(uipc_sim.scene)
    uipc_sim.setup_sim()
    uipc_sim.init_libuipc_scene_rendering()

    camera = setup_camera(sim)
    sim.reset()
    camera.set_world_poses_from_view(
        eyes=torch.tensor([args_cli.camera_eye], device=sim.device),
        targets=torch.tensor([args_cli.camera_target], device=sim.device),
    )
    camera.reset()
    output_path = pathlib.Path(args_cli.output)
    writer = start_video_writer(output_path)

    print(f"[INFO] Recording {args_cli.steps} frames to {output_path}")
    try:
        for _ in range(args_cli.warmup):
            sim.render()
            camera.update(dt=sim.get_physics_dt())

        for step in range(args_cli.steps):
            with Timer("[INFO]: Time taken for uipc sim step", name="uipc_step"):
                uipc_sim.step()
            uipc_sim.update_render_meshes()
            sim.render()
            camera.update(dt=sim.get_physics_dt())
            writer.stdin.write(camera_rgb(camera).tobytes())

            if step % 25 == 0:
                print(f"[INFO] frame {step}/{args_cli.steps}")
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        writer.wait()
        simulation_app.close()

    print(f"[INFO] Saved video: {output_path}")


if __name__ == "__main__":
    main()
