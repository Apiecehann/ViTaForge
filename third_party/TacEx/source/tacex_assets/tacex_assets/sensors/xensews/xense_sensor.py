import os
import torch

from tacex import GelSightSensor


class XenseWSSensor(GelSightSensor):
    """XenseWS-specific tactile sensor depth handling.

    By default this forwards the raw camera depth, matching the base
    GelSightSensor path. Set ``XENSE_USE_BASELINE_FILTER=1`` to enable the older
    no-contact baseline/contact-mask filter for diagnostics.
    """

    def __init__(self, *args, **kwargs):
        self._xense_use_baseline_filter = os.environ.get("XENSE_USE_BASELINE_FILTER", "0") == "1"
        self._xense_baseline_depth_m = None
        self._xense_baseline_margin_m = 0.0005
        self._xense_max_delta_m = 0.0050
        self._xense_debug_depth_count = 0
        super().__init__(*args, **kwargs)

    def reset(self, env_ids=None):
        self._xense_baseline_depth_m = None
        return super().reset(env_ids)

    def _sanitize_depth_m(self):
        """Return sensor-camera depth in meters.

        Raw mode mirrors the base GelSightSensor behavior: pass the camera depth
        through and only replace invalid pixels with the far plane. The optional
        baseline filter keeps the older Xense-specific contact mask available
        behind XENSE_USE_BASELINE_FILTER=1.
        """
        near, far = self.cfg.sensor_camera_cfg.clipping_range
        raw = self.camera.data.output["depth"][:, :, :, 0]
        raw_finite = torch.isfinite(raw)
        raw_in_clip = raw_finite & (raw >= near) & (raw <= far)

        depth = raw.clone()
        depth[~raw_in_clip] = far

        contact_near = self.cfg.optical_sim_cfg.gelpad_to_camera_min_distance - 0.001
        contact_far = (
            self.cfg.optical_sim_cfg.gelpad_to_camera_min_distance
            + self.cfg.gelpad_dimensions.height
        )
        raw_contact_window = raw_in_clip & (depth >= contact_near) & (depth <= contact_far)

        if not self._xense_use_baseline_filter:
            baseline_valid = torch.zeros_like(raw_in_clip, dtype=torch.bool)
            delta = torch.zeros_like(depth)
            self._debug_depth(
                raw, raw_finite, raw_in_clip, baseline_valid, raw_contact_window,
                delta, near, far, contact_near, contact_far
            )
            return depth

        baseline_missing = (
            self._xense_baseline_depth_m is None
            or self._xense_baseline_depth_m.shape != depth.shape
        )
        if baseline_missing:
            if raw_in_clip.any():
                self._xense_baseline_depth_m = depth.detach().clone()
            else:
                # Isaac camera buffers can be all-NaN on the first rendered tactile frame.
                # Do not capture that frame as the baseline, otherwise every later pixel
                # is rejected by baseline_valid and the tactile image stays blank forever.
                baseline = depth
                baseline_valid = torch.zeros_like(raw_in_clip, dtype=torch.bool)
                delta = torch.zeros_like(depth)
                contact = torch.zeros_like(raw_in_clip, dtype=torch.bool)
                self._debug_depth(raw, raw_finite, raw_in_clip, baseline_valid, contact, delta, near, far, contact_near, contact_far)
                depth[:] = far
                return depth

        baseline = self._xense_baseline_depth_m
        delta = baseline - depth

        baseline_valid = baseline < far - 1e-5
        contact = (
            baseline_valid
            & (delta > self._xense_baseline_margin_m)
            & (delta < self._xense_max_delta_m)
            & (depth >= contact_near)
            & (depth <= contact_far)
        )

        self._debug_depth(raw, raw_finite, raw_in_clip, baseline_valid, contact, delta, near, far, contact_near, contact_far)
        depth[~contact] = far

        return depth

    def _debug_depth(self, raw, raw_finite, raw_in_clip, baseline_valid, contact, delta, near, far, contact_near, contact_far):
        if os.environ.get("XENSE_DEBUG_DEPTH", "0") != "1":
            return

        debug_every = max(1, int(os.environ.get("XENSE_DEBUG_DEPTH_EVERY", "1")))
        debug_max = int(os.environ.get("XENSE_DEBUG_DEPTH_MAX", "80"))
        debug_idx = self._xense_debug_depth_count
        self._xense_debug_depth_count += 1

        if debug_idx >= debug_max or debug_idx % debug_every != 0:
            return

        def stat(t):
            return float(t.detach().cpu().item())

        def quantiles(t):
            if t.numel() == 0:
                return {}
            qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=t.device)
            vals = torch.quantile(t.float(), qs)
            return {f"p{int(q.item() * 100):02d}": stat(v) for q, v in zip(qs, vals)}

        raw_finite_values = raw[raw_finite]
        raw_valid = raw[raw_in_clip]
        delta_valid = delta[baseline_valid]
        if raw.shape[-1] > 100 and raw.shape[-2] > 100:
            crop = raw[..., 50:-50, 50:-50]
            crop_finite = torch.isfinite(crop)
            crop_in_clip = crop_finite & (crop >= near) & (crop <= far)
            crop_valid = crop[crop_in_clip]
            crop_contact = contact[..., 50:-50, 50:-50]
        else:
            crop_in_clip = torch.zeros_like(raw_in_clip, dtype=torch.bool)
            crop_valid = raw.new_empty((0,))
            crop_contact = torch.zeros_like(raw_in_clip, dtype=torch.bool)
        msg = {
            "idx": debug_idx,
            "prim": self.cfg.prim_path,
            "near": near,
            "far": far,
            "contact_near": contact_near,
            "contact_far": contact_far,
            "raw_min": stat(raw_finite_values.min()) if raw_finite_values.numel() > 0 else float("nan"),
            "raw_max": stat(raw_finite_values.max()) if raw_finite_values.numel() > 0 else float("nan"),
            "raw_mean": stat(raw_finite_values.mean()) if raw_finite_values.numel() > 0 else float("nan"),
            "raw_finite_frac": stat(raw_finite.float().mean()),
            "raw_in_clip_frac": stat(raw_in_clip.float().mean()),
            "baseline_ready": self._xense_baseline_depth_m is not None,
            "baseline_valid_frac": stat(baseline_valid.float().mean()),
            "contact_frac": stat(contact.float().mean()),
            "crop_in_clip_frac": stat(crop_in_clip.float().mean()) if crop_in_clip.numel() > 0 else float("nan"),
            "crop_contact_frac": stat(crop_contact.float().mean()) if crop_contact.numel() > 0 else float("nan"),
        }
        msg.update({f"raw_{k}": v for k, v in quantiles(raw_finite_values).items()})
        if raw_valid.numel() > 0:
            msg["raw_in_clip_min"] = stat(raw_valid.min())
            msg["raw_in_clip_max"] = stat(raw_valid.max())
            msg["raw_in_clip_mean"] = stat(raw_valid.mean())
            msg.update({f"clip_{k}": v for k, v in quantiles(raw_valid).items()})
        if delta_valid.numel() > 0:
            msg["delta_valid_min"] = stat(delta_valid.min())
            msg["delta_valid_max"] = stat(delta_valid.max())
            msg["delta_valid_mean"] = stat(delta_valid.mean())
            msg.update({f"delta_{k}": v for k, v in quantiles(delta_valid).items()})
        if crop_valid.numel() > 0:
            msg["crop_raw_min"] = stat(crop_valid.min())
            msg["crop_raw_max"] = stat(crop_valid.max())
            msg["crop_raw_mean"] = stat(crop_valid.mean())
            msg.update({f"crop_{k}": v for k, v in quantiles(crop_valid).items()})
        print(f"[xense-depth-debug] {msg}", flush=True)

    def _get_height_map(self):
        if self.camera is None:
            return self._data.output["height_map"]

        depth = self._sanitize_depth_m()
        self._data.output["height_map"][:] = depth
        self._data.output["height_map"] *= 1000.0
        return self._data.output["height_map"]

    def _get_camera_depth(self):
        if self.camera is None:
            return self._data.output["camera_depth"]

        near, far = self.cfg.sensor_camera_cfg.clipping_range
        depth = self._sanitize_depth_m()

        depth_mm = depth.reshape(
            (self._num_envs, 1, self.camera_resolution[1], self.camera_resolution[0])
        ) * 1000.0

        normalized = (depth_mm - near * 1000.0) / max((far - near) * 1000.0, 1e-6)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        normalized = (normalized * 255.0).to(dtype=torch.uint8)

        self._data.output["camera_depth"] = normalized.reshape(
            (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 1)
        )
        return self._data.output["camera_depth"]
