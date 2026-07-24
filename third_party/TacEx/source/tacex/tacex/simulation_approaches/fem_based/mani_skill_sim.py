from __future__ import annotations

import time
import copy
import math
from re import M
import numpy as np
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

import cv2
import omni.usd
import isaaclab.utils.math as math_utils

from tacex_uipc import UipcObject

from ...gelsight_sensor import GelSightSensor
from ..gelsight_simulator import GelSightSimulator
from .sim import VisionTactileSensorUIPC

if TYPE_CHECKING:
    from .mani_skill_sim_cfg import ManiSkillSimulatorCfg

class ManiSkillSimulator(GelSightSimulator):
    """Wrapper for ManiSkill-ViTac simulator for GelSight sensors.

    Instead of IPC, we use UIPC.
    The original ManiSkill-ViTac simulator can be found here https://github.com/chuanyune/ManiSkill-ViTac2025.git
    """

    cfg: ManiSkillSimulatorCfg
    patch_array = None

    def __init__(self, sensor: GelSightSensor, cfg: ManiSkillSimulatorCfg):
        self.sensor: GelSightSensor = sensor

        # needed for VisionTactileSensorUIPC class
        self.camera = None
        self.gelpad_uipc: UipcObject = self.sensor.gelpad_obj
        self.radius = cfg.marker_radius
        self.draw_patch_array()

        super().__init__(sensor=sensor, cfg=cfg)

    def _initialize_impl(self):
        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else:
            self._device = self.cfg.device

        self._num_envs = self.sensor._num_envs

        # todo make size adaptable? I mean with env_ids. This way we would always simulate everything
        self._indentation_depth = torch.zeros((self.sensor._num_envs), device=self.sensor._device)
        """Indentation depth, i.e. how deep the object is pressed into the gelpad.
        Values are in mm.

        Indentation depth is equal to the maximum pressing depth of the object in the gelpad.
        It is used for shifting the height map for the Taxim simulation.
        """

        self.camera = self.sensor.camera
        self.marker_motion_sim: VisionTactileSensorUIPC = VisionTactileSensorUIPC(
            self.gelpad_uipc,
            self.camera,
            sensor_type=self.cfg.sensor_type,
            tactile_img_width=self.cfg.tactile_img_res[0],
            tactile_img_height=self.cfg.tactile_img_res[1],
            marker_shape=self.cfg.marker_shape,
            marker_interval=self.cfg.marker_interval,
            sub_marker_num=self.cfg.sub_marker_num,
            marker_radius=self.radius,
            num_markers=self.cfg.marker_params.num_markers,
            camera_to_surface=self.cfg.camera_to_surface,
            real_size=self.cfg.real_size,
        )

        self.marker_motion_sim._gen_marker_grid()
        self.canonical_marker_uv = self._canonical_marker_uv()
        self._marker_flow_is_visual = False
        self._xsense_visual_contact_weight_baseline = None
        self._xsense_depth_motion_baseline_mm = None
        self._xsense_marker_force_baseline = None

        # create buffers
        self.marker_data = torch.zeros(
            (self.sensor._num_envs, 2, self.cfg.marker_params.num_markers, 2), device=self._device
        )
        """Marker flow data. Shape is [num_envs, 2, num_markers, 2]

        dim=1: [initial, current] marker positions
        dim=3: [x,y] values of the markers
        """

    def marker_motion_simulation(self):
        marker_flow = self.marker_motion_sim.gen_marker_flow()
        self._marker_flow_is_visual = False
        if str(self.cfg.sensor_type).startswith("xense"):
            marker_flow = self._xsense_marker_flow(marker_flow)
            self._marker_flow_is_visual = True
        # todo do it properly for multi env, currently marker flow has shape [2, num_markers, 2] and we want [num_envs, 2, num_markers, 2]
        self.marker_data[0] = torch.as_tensor(
            marker_flow, dtype=self.marker_data.dtype, device=self.marker_data.device
        )
        return self.marker_data

    def _as_marker_tensor(self, value, device: torch.device | str) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().to(device=device, dtype=torch.float32)
        return torch.as_tensor(value, device=device, dtype=torch.float32)

    def _marker_surface_displacement_camera(self, device: torch.device | str) -> torch.Tensor | None:
        marker_sim = self.marker_motion_sim
        try:
            marker_idx = torch.as_tensor(marker_sim.marker_surf_idx, device=device, dtype=torch.long)
            marker_weight = torch.as_tensor(marker_sim.marker_weight, device=device, dtype=torch.float32)
            reference = self._as_marker_tensor(marker_sim.reference_surface_vertices_camera, device)
            current = self._as_marker_tensor(marker_sim.get_surface_vertices_camera(), device)
            ref_pts = (reference[marker_idx] * marker_weight[..., None]).sum(1)
            curr_pts = (current[marker_idx] * marker_weight[..., None]).sum(1)

            constrain_ids = torch.as_tensor(marker_sim.constrain_ids, device=device, dtype=torch.long)
            if constrain_ids.numel() > 0:
                vertices = self._as_marker_tensor(marker_sim.get_vertices_camera(), device)
                constrain_pts = self._as_marker_tensor(marker_sim.constrain_pts, device)
                mean_motion = (vertices[constrain_ids] - constrain_pts).mean(dim=0)
                curr_pts[:, :2] -= mean_motion[:2]
            return curr_pts - ref_pts
        except Exception:
            return None

    def _marker_contact_force_sensor(self, device: torch.device | str) -> torch.Tensor | None:
        if not bool(getattr(self.cfg, "marker_motion_force_enabled", False)):
            return None

        uipc_sim = getattr(self.gelpad_uipc, "uipc_sim", None)
        if uipc_sim is None or not hasattr(uipc_sim, "get_contact_gradient"):
            return None

        marker_sim = self.marker_motion_sim
        try:
            idx, grad = uipc_sim.get_contact_gradient()
            nodal_pos = self.gelpad_uipc.data.nodal_pos_w
            force_device = nodal_pos.device
            idx = torch.as_tensor(idx, device=force_device, dtype=torch.long)
            grad = torch.as_tensor(grad, device=force_device, dtype=torch.float32)
            offsets = uipc_sim._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"]
            start = int(offsets[self.gelpad_uipc.global_system_id])
            num_v = int(nodal_pos.shape[0])
            dense = torch.zeros((num_v, 3), dtype=torch.float32, device=force_device)
            if idx.numel() > 0:
                mask = (idx >= start) & (idx < start + num_v)
                if bool(mask.any()):
                    dense[idx[mask] - start] = -grad[mask]

            surf_global = torch.as_tensor(marker_sim.vertices_on_surface, device=force_device, dtype=torch.long)
            marker_idx = torch.as_tensor(marker_sim.marker_surf_idx, device=force_device, dtype=torch.long)
            marker_weight = torch.as_tensor(marker_sim.marker_weight, device=force_device, dtype=torch.float32)
            force_surf = dense[surf_global]
            marker_force = (force_surf[marker_idx] * marker_weight[..., None]).sum(1)

            self.camera._update_poses(self.camera._ALL_INDICES)
            rot_w_sensor = math_utils.matrix_from_quat(self.camera._data.quat_w_ros)[0]
            marker_force = marker_force @ rot_w_sensor.to(device=force_device, dtype=marker_force.dtype)
            return marker_force.to(device=device, dtype=torch.float32)
        except Exception:
            return None

    def _radial_marker_direction(self, marker_uv: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        if weight is None or weight.numel() == 0 or float(weight.sum().item()) <= 1.0e-8:
            center = marker_uv.mean(dim=0, keepdim=True)
        else:
            center = (marker_uv * weight[:, None]).sum(dim=0, keepdim=True) / weight.sum().clamp_min(1.0e-8)
        radial = marker_uv - center
        norm = torch.linalg.norm(radial, dim=-1, keepdim=True).clamp_min(1.0e-6)
        return torch.nan_to_num(radial / norm)

    def _xsense_subtract_static_baseline(
        self,
        value: torch.Tensor | None,
        attr_name: str,
        count: int,
        clamp_min: float | None = 0.0,
        clamp_max: float | None = None,
    ) -> torch.Tensor | None:
        if value is None or value.shape[0] < count:
            return value

        current = torch.nan_to_num(value[:count]).detach()
        baseline = getattr(self, attr_name, None)
        if baseline is None or baseline.shape[0] != count:
            setattr(self, attr_name, current.clone())
            return torch.zeros_like(current)

        baseline = baseline.to(device=current.device, dtype=current.dtype)
        corrected = current - baseline[:count]
        if clamp_min is not None:
            corrected = corrected.clamp_min(clamp_min)
        if clamp_max is not None:
            corrected = corrected.clamp(max=clamp_max)
        return corrected

    def _xsense_marker_flow(self, raw_marker_flow) -> torch.Tensor:
        device = self.canonical_marker_uv.device
        raw_marker_flow = self._as_marker_tensor(raw_marker_flow, device)
        num_markers = int(self.cfg.marker_params.num_markers)
        if raw_marker_flow.shape[1] != num_markers or self.canonical_marker_uv.shape[0] != num_markers:
            raise RuntimeError(
                "XSense FEM marker count does not match the canonical grid: "
                f"raw={raw_marker_flow.shape[1]}, canonical={self.canonical_marker_uv.shape[0]}, "
                f"expected={num_markers}"
            )

        canonical_uv = self.canonical_marker_uv.to(device=device, dtype=torch.float32)
        scale = float(getattr(self.cfg, "marker_visual_motion_scale", 1.0))
        fem_delta_uv = torch.nan_to_num(raw_marker_flow[1] - raw_marker_flow[0])
        return torch.stack(
            (canonical_uv, canonical_uv + scale * fem_delta_uv),
            dim=0,
        )

    def _canonical_marker_uv(self) -> torch.Tensor:
        sx, sy = self.cfg.marker_shape
        width, height = self.cfg.tactile_img_res
        bounds = getattr(self.cfg, "marker_visual_bounds", None)
        if bounds is None:
            margin_x = 0.135 * width
            margin_y = 0.235 * height
            x_min, x_max = margin_x, width - margin_x
            y_min, y_max = margin_y, height - margin_y
        else:
            x_min, y_min, x_max, y_max = bounds
            x_min, x_max = x_min * width, x_max * width
            y_min, y_max = y_min * height, y_max * height
        xs = torch.linspace(x_min, x_max, sx, device=self._device)
        ys = torch.linspace(y_min, y_max, sy, device=self._device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        marker_uv = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        jitter_ratio = getattr(self.cfg, "marker_visual_jitter", 0.0)
        if jitter_ratio > 0:
            num_markers = marker_uv.shape[0]
            idx = torch.arange(num_markers, device=self._device, dtype=marker_uv.dtype)
            seed = float(getattr(self.cfg, "marker_visual_jitter_seed", 20260623))
            rand_x = torch.remainder(torch.sin(idx * 12.9898 + seed * 0.001) * 43758.5453, 1.0)
            rand_y = torch.remainder(torch.sin(idx * 78.233 + seed * 0.002) * 24634.6345, 1.0)
            step_x = (x_max - x_min) / max(sx - 1, 1)
            step_y = (y_max - y_min) / max(sy - 1, 1)
            jitter = torch.stack(
                ((rand_x - 0.5) * step_x * jitter_ratio, (rand_y - 0.5) * step_y * jitter_ratio),
                dim=-1,
            )
            marker_uv = marker_uv + jitter
            pad = self.radius + 1.0
            marker_uv[:, 0] = marker_uv[:, 0].clamp(pad, width - pad)
            marker_uv[:, 1] = marker_uv[:, 1].clamp(pad, height - pad)
        return marker_uv[: self.cfg.marker_params.num_markers]

    def _smooth_marker_delta_for_visual(self, delta: torch.Tensor) -> torch.Tensor:
        passes = int(getattr(self.cfg, "marker_visual_flow_smoothing", 0))
        if passes <= 0:
            return delta

        sx, sy = self.cfg.marker_shape
        grid_count = int(sx * sy)
        if delta.shape[0] < grid_count:
            return delta

        head = delta[:grid_count]
        tail = delta[grid_count:]
        grid = head.reshape(sy, sx, 2).permute(2, 0, 1).unsqueeze(0)
        for _ in range(passes):
            grid = F.avg_pool2d(F.pad(grid, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        smoothed = grid.squeeze(0).permute(1, 2, 0).reshape(grid_count, 2)
        if tail.numel() == 0:
            return smoothed
        return torch.cat((smoothed, tail), dim=0)

    def _sample_visual_contact_weight(self, marker_uv: torch.Tensor) -> torch.Tensor | None:
        weight_map = getattr(self.sensor, "_surface_deformation_marker_contact_weight", None)
        if weight_map is None:
            weight_map = getattr(self.sensor, "_surface_deformation_contact_weight", None)
        if weight_map is None:
            return None
        weight_map = torch.as_tensor(weight_map, dtype=torch.float32, device=marker_uv.device)
        if weight_map.ndim == 3:
            weight_map = weight_map[0]
        if weight_map.ndim != 2 or marker_uv.ndim != 2 or marker_uv.shape[-1] != 2:
            return None

        height, width = weight_map.shape
        work = weight_map.view(1, 1, height, width)
        if (width, height) != tuple(self.cfg.tactile_img_res):
            work = F.interpolate(
                work,
                size=(self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]),
                mode="bilinear",
                align_corners=False,
            )
            height, width = work.shape[-2:]

        denom_x = max(float(width - 1), 1.0)
        denom_y = max(float(height - 1), 1.0)
        grid = torch.empty((1, marker_uv.shape[0], 1, 2), dtype=torch.float32, device=marker_uv.device)
        grid[0, :, 0, 0] = marker_uv[:, 0] / denom_x * 2.0 - 1.0
        grid[0, :, 0, 1] = marker_uv[:, 1] / denom_y * 2.0 - 1.0
        sampled = F.grid_sample(work, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return sampled[0, 0, :, 0].clamp(0.0, 1.0)

    def _remove_background_visual_drift(self, delta: torch.Tensor, contact_weight: torch.Tensor) -> torch.Tensor:
        if delta.ndim != 2 or delta.shape[-1] != 2 or contact_weight.ndim != 1 or contact_weight.shape[0] != delta.shape[0]:
            return delta - delta.mean(dim=0, keepdim=True)

        threshold = float(getattr(self.cfg, "marker_visual_background_threshold", 0.18) or 0.18)
        bg_mask = contact_weight <= threshold
        min_bg = max(6, delta.shape[0] // 5)
        if int(bg_mask.sum().item()) < min_bg:
            quantile = torch.quantile(contact_weight, 0.35)
            bg_mask = contact_weight <= quantile
        if int(bg_mask.sum().item()) <= 0:
            return delta - delta.mean(dim=0, keepdim=True)
        return delta - delta[bg_mask].mean(dim=0, keepdim=True)

    def marker_rgb_motion(self) -> torch.Tensor:
        return self.marker_data[0, 1]

    def reset(self):
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        self._xsense_visual_contact_weight_baseline = None
        self._xsense_depth_motion_baseline_mm = None
        self._xsense_marker_force_baseline = None
        # self.init_marker_pos = (self.marker_motion_sim.init_marker_x_pos, self.marker_motion_sim.init_marker_y_pos)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Creates an USD attribute for the sensor asset, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".
        There is an attribute called "show_tactile_image".
        Toggle it on to show the sensor output in the GUI.

        If only optical simulation is used, then only an optical img is displayed.
        If only the marker simulatios is used, then only an image displaying the marker positions is displayed.
        If both, optical and marker simulation, are used, then the images are overlaid.
        """
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "_debug_windows"):
                # dict of windows that show the simulated tactile images, if the attribute of the sensor asset is turned on
                self._debug_windows = {}
                self._debug_img_providers = {}
                # todo check if we can make implementation more efficient than dict of dicts
                if "marker_motion" in self.sensor.cfg.data_types:
                    self._debug_windows = {}
                    self._debug_img_providers = {}
        else:
            pass

    def _debug_vis_callback(self, event):
        if self.sensor._prim_view is None:
            return

        # Update the GUI windows_prim_view
        for i, prim in enumerate(self.sensor._prim_view.prims):
            if "marker_motion" in self.sensor.cfg.data_types:
                show_img = prim.GetAttribute("debug_marker_motion").Get()
                if show_img:
                    if str(i) not in self._debug_windows:
                        # create a window
                        window = omni.ui.Window(
                            self.sensor._prim_view.prim_paths[i] + "/fem_marker",
                            width=self.cfg.tactile_img_res[0],
                            height=self.cfg.tactile_img_res[1],
                        )
                        self._debug_windows[str(i)] = window
                        # create image provider
                        self._debug_img_providers[str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    tactile_rgb = self.sensor.data.output["tactile_rgb"][i] / 255.0
                    marker_motion = self.marker_rgb_motion()
                    marker_img = self.draw_markers(marker_uv=marker_motion)
                    tactile_rgb *= torch.dstack([marker_img / 255] * 3)
                    frame = (tactile_rgb * 255).to(dtype=torch.uint8).cpu().numpy()
                    # marker_flow_i = self.sensor.data.output["marker_motion"][i]

                    # frame = self._create_marker_img(marker_flow_i)
                    # draw current marker positions like ManiSkill-ViTac does
                    # frame = self.draw_markers(
                    #     marker_flow_i[1].cpu().numpy(),
                    #     img_w=self.cfg.tactile_img_res[0],
                    #     img_h=self.cfg.tactile_img_res[1],
                    # )

                    # create tactile rgb img with markers
                    # if "tactile_rgb" in self.sensor.cfg.data_types:
                    #     if (
                    #         self.sensor.cfg.optical_sim_cfg.tactile_img_res
                    #         == self.sensor.cfg.marker_motion_sim_cfg.tactile_img_res
                    #     ):
                    #         # todo add upscaling of tactile_rgb, if not same size
                    #         tactile_rgb = self.sensor.data.output["tactile_rgb"][i].cpu().numpy() * 255
                    #         frame = tactile_rgb * np.dstack([frame.astype(np.float64) / 255] * 3)

                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

                    height, width, channels = frame.shape

                    with self._debug_windows[str(i)].frame:
                        self._debug_img_providers[str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._debug_img_providers[str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._debug_windows:
                    # remove window/img_provider from dictionary and destroy them
                    self._debug_windows.pop(str(i)).destroy()
                    self._debug_img_providers.pop(str(i)).destroy()

    def _create_marker_img(self, marker_data):
        """Visualization of marker flow like in the original FOTS simulation.

        Marker data needs to have the shape [2, num_markers, 2]
        - dim=0: init and current markers
        - dim=2: x and y values of the marker position

        Args:
            marker_data: marker flow data with shape [2, num_markers, 2]
        """
        # for visualization -> white background with black dots
        color = (0, 0, 0)
        arrow_scale = 1  # 10 #0.0001 #0.25

        frame = np.ones((self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0])).astype(np.uint8)

        # marker data has shape [2, num_markers, 2], where first dim = init and current marker position
        init_marker_pos = marker_data[0].cpu().numpy()
        current_marker_pos = marker_data[1].cpu().numpy()

        num_markers = marker_data.shape[1]
        for marker_index in range(num_markers):
            init_x_pos = int(init_marker_pos[marker_index][0])
            init_y_pos = int(init_marker_pos[marker_index][1])

            x_pos = int(current_marker_pos[marker_index][0])
            y_pos = int(current_marker_pos[marker_index][1])

            if (x_pos >= frame.shape[1]) or (x_pos < 0) or (y_pos >= frame.shape[0]) or (y_pos < 0):
                continue
            # cv2.circle(frame,(column,row), 6, (255,255,255), 1, lineType=8)

            pt1 = (init_x_pos, init_y_pos)
            pt2 = (x_pos + arrow_scale * int(x_pos - init_x_pos), y_pos + arrow_scale * int(y_pos - init_y_pos))

            cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.2)

        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return frame

    def draw_patch_array(self, super_resolution_ratio:int=10):
        if ManiSkillSimulator.patch_array is not None:
            return
        
        circle_radius = self.radius
        size_slot_num = 50
        base_circle_radius = circle_radius * 0.5  # 与原始实现对齐: 1.5 / 3 = 0.5 比例
        blur_k, blur_sigma = (17, 17), 15 
        
        # patch 尺寸为 4 * circle_radius，用于覆盖 [-2r, 2r] 范围
        patch_size = 4 * circle_radius
        patch_array = np.zeros(
            (
                super_resolution_ratio,
                super_resolution_ratio,
                size_slot_num,
                patch_size,
                patch_size,
            ),
            dtype=np.uint8,
        )
        for u in range(super_resolution_ratio):
            for v in range(super_resolution_ratio):
                for w in range(size_slot_num):
                    # 高分辨率图像用于亚像素渲染
                    img_highres = (
                        np.ones(
                            (
                                patch_size * super_resolution_ratio,
                                patch_size * super_resolution_ratio,
                            ),
                            dtype=np.uint8,
                        )
                        * 255
                    )
                    # 圆心在高分辨率图像中心
                    center = np.array(
                        [
                            circle_radius * super_resolution_ratio * 2,
                            circle_radius * super_resolution_ratio * 2,
                        ],
                        dtype=np.int32,
                    )
                    # 亚像素偏移: u, v 控制圆心的亚像素位置
                    center_offseted = center + np.array([u, v])
                    # w 控制圆的大小变化
                    draw_radius = round(base_circle_radius * super_resolution_ratio + w)
                    img_highres = cv2.circle(
                        img_highres,
                        tuple(center_offseted),
                        draw_radius,
                        (0, 0, 0),
                        thickness=cv2.FILLED,
                        lineType=cv2.LINE_AA,
                    )
                    img_highres = cv2.GaussianBlur(img_highres, blur_k, blur_sigma)
                    # 下采样到目标尺寸
                    img_lowres = cv2.resize(
                        img_highres,
                        (patch_size, patch_size),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    patch_array[u, v, w, ...] = img_lowres

        ManiSkillSimulator.patch_array = torch.tensor(
            patch_array, dtype=torch.uint8, device="cuda:0"
        )
        # 保存参数供 draw_markers 使用
        ManiSkillSimulator.patch_params = {
            "circle_radius": circle_radius,
            "base_circle_radius": base_circle_radius,
            "super_resolution_ratio": super_resolution_ratio,
        }

    def draw_markers(self, marker_uv: torch.Tensor) -> torch.Tensor:
        """Visualize the marker flow like the ManiSkill-ViTac Simulator does.

        Reference:
        https://github.com/chuanyune/ManiSkill-ViTac2025/blob/a3d7df54bca9a2e57f34b37be3a3df36dc218915/Track_1/envs/tactile_sensor_sapienipc.py

        Args:
            marker_uv: Marker flow of a sensor. Shape is (2, num_markers, 2).
            marker_size: The size of the markers in the image. Defaults to 3.
            img_w: Width of the tactile image. Defaults to 320.
            img_h: Height of the tactile image. Defaults to 240.

        Returns:
            Image with the markers visualized as dots.
        """
        if str(self.cfg.sensor_type).startswith("xense") and hasattr(self, "canonical_marker_uv"):
            marker_uv = self.marker_rgb_motion()

        device = "cuda:0"
        params = ManiSkillSimulator.patch_params
        circle_radius = params["circle_radius"]
        base_circle_radius = params["base_circle_radius"]
        super_resolution_ratio = params["super_resolution_ratio"]
        
        pad_size = 2 * circle_radius
        patch_size = 4 * circle_radius
        
        marker_image = torch.ones((self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]),
                               dtype=torch.float32, device=device) * 255
        marker_image = torch.nn.functional.pad(
            marker_image, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=255)
        
        uvs = marker_uv + 0.5 + pad_size
        u_floor = torch.floor(uvs[:, 0]).long()
        v_floor = torch.floor(uvs[:, 1]).long()
        u_frac = uvs[:, 0] - u_floor
        v_frac = uvs[:, 1] - v_floor
        
        patch_id_u = torch.floor(u_frac * super_resolution_ratio).long().clamp(0, super_resolution_ratio - 1)
        patch_id_v = torch.floor(v_frac * super_resolution_ratio).long().clamp(0, super_resolution_ratio - 1)
        marker_size = circle_radius
        patch_id_w = int((marker_size - base_circle_radius) * super_resolution_ratio)
        patch_id_w = max(0, min(patch_id_w, 49))  # clamp to [0, size_slot_num - 1]

        patches = ManiSkillSimulator.patch_array[patch_id_u, patch_id_v, patch_id_w, :, :]
        
        for i in range(len(patches)):
            u_start = u_floor[i].item() - pad_size
            v_start = v_floor[i].item() - pad_size
            if marker_image.shape[1] - patch_size > u_start >= 0 and marker_image.shape[0] - patch_size > v_start >= 0:
                replace_idx = (
                    slice(v_start, v_start + patch_size),
                    slice(u_start, u_start + patch_size)
                )
                old_status = marker_image[replace_idx]
                new_status = torch.minimum(old_status, patches[i].float())
                marker_image[replace_idx] = new_status

        marker_image = marker_image[pad_size:-pad_size, pad_size:-pad_size]
        
        noise_level = float(getattr(self.cfg, "marker_visual_noise", 80.0) or 0.0)
        if noise_level > 0:
            noise = torch.rand_like(marker_image) * noise_level
            marker_mask = marker_image < 255
            marker_image = torch.where(
                marker_mask,
                torch.clamp(marker_image + noise, 0, 255),
                marker_image)
        
        return marker_image
