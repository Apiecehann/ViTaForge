from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import omni.usd
import torch.nn.functional as torch_F
import torchvision.transforms.functional as F

from ...gelsight_sensor import GelSightSensor
from ..gelsight_simulator import GelSightSimulator
from .sim import Taxim

if TYPE_CHECKING:
    from .taxim_sim_cfg import TaximSimulatorCfg


class TaximSimulator(GelSightSimulator):
    """Wraps around the Taxim simulation for the optical simulation of GelSight sensors
    inside Isaac Sim.

    """

    cfg: TaximSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: TaximSimulatorCfg):
        self.sensor = sensor

        super().__init__(sensor=sensor, cfg=cfg)

    def _initialize_impl(self):
        calib_folder = Path(self.cfg.calib_folder_path)

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
        self.tactile_rgb_img = torch.zeros(
            (self.sensor._num_envs, self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0], 3),
            device=self._device,
        )

        self._taxim: Taxim = Taxim(calib_folder=calib_folder, device=self._device)
        self._maybe_override_background_image()
        # update Taxim settings via settings from cfg class
        # print(self._taxim.width)
        # self._taxim.width = self.cfg.tactile_img_res[0]
        # self._taxim.height = self.cfg.tactile_img_res[1]

        # -- note Taxim sim uses (channels, height, width) format

        # tactile rgb image without indentation
        self.background_img = self._taxim.background_img
        #  up/downscale height map if different than tactile img res
        if self.background_img.shape != (3, self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            self.background_img = F.resize(
                self.background_img, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0])
            )
        # last dim should be channels for isaac
        self.background_img = self.background_img.movedim(0, 2)

        # use background as initial tactile_rgb_img
        self.tactile_rgb_img[:] = self.background_img

        # if camera resolution is different than the tactile RGB res, scale img
        self.img_res = self.cfg.tactile_img_res

    def _maybe_override_background_image(self) -> None:
        override_path = Path(getattr(self.cfg, "background_img_override_path", "") or "")
        if not override_path.is_file():
            return

        taxim = self._taxim
        np_img_to_torch = getattr(taxim, "_TaximTorch__np_img_to_torch", None)
        bgr_to_rgb = getattr(taxim, "_TaximTorch__bgr_to_rgb", None)
        process_initial_frame = getattr(taxim, "_TaximTorch__process_initial_frame", None)
        background_cache = getattr(taxim, "_TaximTorch__get_background_img_cached", None)
        if (
            np_img_to_torch is None
            or bgr_to_rgb is None
            or process_initial_frame is None
            or background_cache is None
        ):
            return

        override_bgr = cv2.imread(str(override_path), cv2.IMREAD_COLOR)
        if override_bgr is None:
            return

        try:
            override_rgb = bgr_to_rgb(np_img_to_torch(override_bgr / 255.0))
            if bool(getattr(self.cfg, "background_img_override_raw", False)):
                taxim._TaximTorch__bg_proc = override_rgb
            else:
                taxim._TaximTorch__bg_proc = process_initial_frame(override_rgb)
            if hasattr(background_cache, "cache_clear"):
                background_cache.cache_clear()
        except Exception as exc:
            print(f"[TaximSimulator] Warning: failed to apply background override {override_path}: {exc}")

    def _is_xsense_sensor(self) -> bool:
        marker_cfg = getattr(getattr(self.sensor, "cfg", None), "marker_motion_sim_cfg", None)
        sensor_type = str(getattr(marker_cfg, "sensor_type", "")).lower()
        return sensor_type.startswith("xense")

    def _blur_chw(self, value: torch.Tensor, passes: int, kernel_size: int = 5) -> torch.Tensor:
        if passes <= 0:
            return value
        kernel_size = max(int(kernel_size), 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        work = value
        for _ in range(int(passes)):
            work = torch_F.avg_pool2d(
                torch_F.pad(work, (pad, pad, pad, pad), mode="replicate"),
                kernel_size=kernel_size,
                stride=1,
            )
        return work

    def _xsense_contact_gate(self, height_map: torch.Tensor) -> torch.Tensor | None:
        press_depth = torch.as_tensor(
            self._indentation_depth,
            dtype=height_map.dtype,
            device=height_map.device,
        ).flatten()
        if press_depth.numel() == 1 and height_map.shape[0] > 1:
            press_depth = press_depth.repeat(height_map.shape[0])
        if press_depth.numel() < height_map.shape[0]:
            return None

        shifted = (
            height_map
            - height_map.amin(dim=(-2, -1), keepdim=True)
            - press_depth[: height_map.shape[0]].view(-1, 1, 1)
        )
        contact_depth = torch.clamp(-shifted, min=0.0)
        peak = contact_depth.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        gate = (contact_depth / peak).clamp(0.0, 1.0)
        return gate.unsqueeze(1)

    def _process_xsense_contact_gate(
        self,
        gate: torch.Tensor | None,
        target_shape: tuple[int, int],
    ) -> torch.Tensor | None:
        if gate is None:
            return None

        if gate.shape[-2:] != target_shape:
            gate = torch_F.interpolate(gate, size=target_shape, mode="bilinear", align_corners=False)
        threshold = min(
            max(float(getattr(self.cfg, "xsense_response_contact_gate_threshold", 0.05) or 0.0), 0.0),
            0.95,
        )
        gate = torch.clamp((gate - threshold) / max(1.0 - threshold, 1.0e-6), min=0.0, max=1.0)
        gamma = max(float(getattr(self.cfg, "xsense_response_contact_gate_gamma", 1.0) or 1.0), 0.25)
        gate = gate.pow(gamma)
        blur_passes = max(int(getattr(self.cfg, "xsense_response_contact_gate_blur_passes", 1) or 0), 0)
        gate = self._blur_chw(gate, blur_passes, kernel_size=5)
        peak = gate.amax(dim=(-2, -1), keepdim=True)
        gate = torch.where(peak > 1.0e-6, gate / peak.clamp_min(1.0e-6), gate)
        return gate.clamp(0.0, 1.0)

    def _xsense_background_chw(self, rendered_chw: torch.Tensor) -> torch.Tensor:
        background_hwc = self.background_img.to(device=rendered_chw.device, dtype=rendered_chw.dtype)
        background_chw = background_hwc.movedim(2, 0).unsqueeze(0)
        if background_chw.shape[-2:] != rendered_chw.shape[-2:]:
            background_chw = torch_F.interpolate(
                background_chw,
                size=rendered_chw.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if background_chw.shape[0] == 1 and rendered_chw.shape[0] > 1:
            background_chw = background_chw.repeat(rendered_chw.shape[0], 1, 1, 1)
        return background_chw

    def _apply_xsense_taxim_residual_response(
        self,
        rendered_chw: torch.Tensor,
        height_map: torch.Tensor,
        background_chw: torch.Tensor,
    ) -> torch.Tensor:
        residual = rendered_chw - background_chw
        sigma_px = float(getattr(self.cfg, "xsense_response_highpass_sigma_px", 0.0) or 0.0)
        if sigma_px > 0.0:
            kernel = max(3, int(round(sigma_px * 4.0)) | 1)
            pad = kernel // 2
            low = torch_F.avg_pool2d(
                torch_F.pad(residual, (pad, pad, pad, pad), mode="replicate"),
                kernel_size=kernel,
                stride=1,
            )
            residual = residual - low

        gate = self._process_xsense_contact_gate(self._xsense_contact_gate(height_map), rendered_chw.shape[-2:])
        if gate is not None:
            residual = residual * gate

        gain = float(getattr(self.cfg, "xsense_response_residual_gain", 1.0) or 1.0)
        return torch.clamp(background_chw + residual * gain, 0.0, 1.0)

    def _xsense_indent_from_height_map(
        self,
        height_map: torch.Tensor,
        target_shape: tuple[int, int],
    ) -> torch.Tensor:
        baseline = height_map.amax(dim=(-2, -1), keepdim=True)
        indent = (baseline - height_map).clamp_min(0.0).unsqueeze(1)
        if indent.shape[-2:] != target_shape:
            indent = torch_F.interpolate(indent, size=target_shape, mode="bilinear", align_corners=False)
        return indent

    def _apply_xsense_analytic_response(
        self,
        rendered_chw: torch.Tensor,
        height_map: torch.Tensor,
        background_chw: torch.Tensor,
    ) -> torch.Tensor:
        target_shape = rendered_chw.shape[-2:]
        indent = self._xsense_indent_from_height_map(height_map, target_shape)
        peak = indent.amax(dim=(-2, -1), keepdim=True)
        indent_norm = torch.where(peak > 1.0e-6, indent / peak.clamp_min(1.0e-6), torch.zeros_like(indent))

        gate = self._process_xsense_contact_gate(self._xsense_contact_gate(height_map), target_shape)
        if gate is None:
            gate = indent_norm
        if gate.shape[0] == 1 and rendered_chw.shape[0] > 1:
            gate = gate.repeat(rendered_chw.shape[0], 1, 1, 1)
        if float(gate.amax().item()) <= 1.0e-6:
            return background_chw

        indent_gamma = max(float(getattr(self.cfg, "xsense_response_indent_gamma", 0.85) or 0.85), 0.25)
        indent_support = min(
            max(float(getattr(self.cfg, "xsense_response_indent_support", 0.35) or 0.0), 0.0),
            1.0,
        )
        contact_weight = torch.maximum(gate, indent_support * indent_norm.pow(indent_gamma)).clamp(0.0, 1.0)

        padded_indent = torch_F.pad(indent_norm, (1, 1, 1, 1), mode="replicate")
        grad_x = 0.5 * (padded_indent[..., 1:-1, 2:] - padded_indent[..., 1:-1, :-2])
        grad_y = 0.5 * (padded_indent[..., 2:, 1:-1] - padded_indent[..., :-2, 1:-1])
        edge = torch.sqrt(grad_x.square() + grad_y.square())
        edge_flat = edge.flatten(start_dim=1)
        edge_scale = torch.quantile(edge_flat, 0.98, dim=1).view(-1, 1, 1, 1).clamp_min(1.0e-6)
        edge = torch.clamp(edge / edge_scale, 0.0, 1.0) * gate

        contact_rgb = torch.as_tensor(
            getattr(self.cfg, "xsense_response_contact_rgb", (-0.052, -0.002, 0.072)),
            dtype=rendered_chw.dtype,
            device=rendered_chw.device,
        ).view(1, 3, 1, 1)
        edge_rgb = torch.as_tensor(
            getattr(self.cfg, "xsense_response_edge_rgb", (-0.010, 0.0, 0.016)),
            dtype=rendered_chw.dtype,
            device=rendered_chw.device,
        ).view(1, 3, 1, 1)
        edge_gain = float(getattr(self.cfg, "xsense_response_edge_gain", 1.0) or 0.0)

        residual = contact_weight * contact_rgb + edge * edge_rgb * edge_gain
        taxim_mix = min(max(float(getattr(self.cfg, "xsense_response_taxim_residual_mix", 0.0) or 0.0), 0.0), 1.0)
        if taxim_mix > 0.0:
            taxim_img = self._apply_xsense_taxim_residual_response(rendered_chw, height_map, background_chw)
            residual = residual + (taxim_img - background_chw) * taxim_mix

        gain = float(getattr(self.cfg, "xsense_response_residual_gain", 1.0) or 1.0)
        return torch.clamp(background_chw + residual * gain, 0.0, 1.0)

    def _apply_xsense_response(self, rendered_chw: torch.Tensor, height_map: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self.cfg, "xsense_response_enabled", False)):
            return rendered_chw
        if not self._is_xsense_sensor():
            return rendered_chw

        background_chw = self._xsense_background_chw(rendered_chw)
        model = str(getattr(self.cfg, "xsense_response_model", "taxim_residual") or "taxim_residual").lower()
        if model in {"analytic", "analytic_xsense", "calibrated_xsense", "xsense"}:
            return self._apply_xsense_analytic_response(rendered_chw, height_map, background_chw)
        return self._apply_xsense_taxim_residual_response(rendered_chw, height_map, background_chw)

    def optical_simulation(self):
        """Returns simulation output of Taxim optical simulation.

        Images have the shape (num_envs, height, width, channels) and values in range [0,255].
        """
        height_map = self.sensor._data.output["height_map"]

        # up/downscale height map if camera res different than tactile img res
        if (height_map.shape[1], height_map.shape[2]) != (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            height_map = F.resize(height_map, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]))

        if self._device == "cpu":
            height_map = height_map.cpu()

        # only simulate in case of indentation
        # self.tactile_rgb_img[self._indentation_depth <= 0][:] = self.background_img
        # if height_map[self._indentation_depth > 0].shape[0] > 0:
        #     self.tactile_rgb_img[self._indentation_depth > 0] = self._taxim.render_direct(
        #         height_map[self._indentation_depth > 0],
        #         with_shadow=self.cfg.with_shadow,
        #         press_depth=self._indentation_depth[self._indentation_depth > 0],
        #         orig_hm_fmt=False,
        #     ).movedim(1, 3) #*255).type(torch.uint8)

        rendered = self._taxim.render_direct(
            height_map[:],
            with_shadow=self.cfg.with_shadow,
            press_depth=self._indentation_depth,
            orig_hm_fmt=False,
        )
        rendered = self._apply_xsense_response(rendered, height_map)
        self.tactile_rgb_img[:] = rendered.movedim(1, 3)  # *255).type(torch.uint8)

        return self.tactile_rgb_img

    def compute_indentation_depth(self):
        height_map = self.sensor._data.output["height_map"] / 1000  # convert height map from mm to meter
        min_distance_obj = height_map.amin((1, 2))
        # smallest distance between object and sensor case
        dist_obj_sensor_case = min_distance_obj - self.cfg.gelpad_to_camera_min_distance

        # print("dist_obj_sensor_case", dist_obj_sensor_case)
        # if (dist_obj_sensor_case < 0):  # object is "inside the sensor", cause the object is closer to the camera than the edge of the sensor
        #     # print("Object is inside the sensor!!! Gelpad would be broken!!!")
        #     dist_obj_sensor_case = 0
        dist_obj_sensor_case = torch.where(dist_obj_sensor_case < 0, 0, dist_obj_sensor_case)

        self._indentation_depth[:] = torch.where(
            dist_obj_sensor_case <= self.cfg.gelpad_height, (self.cfg.gelpad_height - dist_obj_sensor_case) * 1000, 0
        )

        return self._indentation_depth

    def reset(self):
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        self.tactile_rgb_img[:] = self.background_img

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
                if "tactile_rgb" in self.sensor.cfg.data_types:
                    self._debug_windows = {}
                    self._debug_img_providers = {}
        else:
            pass

    def _debug_vis_callback(self, event):
        if self.sensor._prim_view is None:
            return

        # Update the GUI windows
        for i, prim in enumerate(self.sensor.prim_view.prims):
            if "tactile_rgb" in self.sensor.cfg.data_types:
                show_img = prim.GetAttribute("debug_tactile_rgb").Get()
                if show_img:
                    if str(i) not in self._debug_windows:
                        # create a window
                        window = omni.ui.Window(
                            self.sensor._prim_view.prim_paths[i] + "/taxim_rgb",
                            height=self.cfg.tactile_img_res[1],
                            width=self.cfg.tactile_img_res[0],
                        )
                        self._debug_windows[str(i)] = window
                        # create image provider
                        self._debug_img_providers[str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self.sensor.data.output["tactile_rgb"][i].cpu().numpy() * 255
                    frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # update image of the window
                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)  # cv.COLOR_BGR2RGBA) COLOR_RGB2RGBA
                    height, width, channels = frame.shape

                    with self._debug_windows[str(i)].frame:
                        # self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
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
