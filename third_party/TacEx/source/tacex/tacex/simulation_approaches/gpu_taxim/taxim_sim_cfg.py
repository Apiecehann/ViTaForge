from dataclasses import MISSING

from isaaclab.utils import configclass

from ..gelsight_simulator_cfg import GelSightSimulatorCfg
from .taxim_sim import TaximSimulator

"""Configuration for a tactile RGB simulation with Taxim."""


@configclass
class TaximSimulatorCfg(GelSightSimulatorCfg):
    simulation_approach_class: type = TaximSimulator

    calib_folder_path: str = ""
    background_img_override_path: str = ""
    background_img_override_raw: bool = False

    xsense_response_enabled: bool = False
    xsense_response_model: str = "taxim_residual"
    xsense_response_residual_gain: float = 1.0
    xsense_response_highpass_sigma_px: float = 0.0
    xsense_response_contact_gate_threshold: float = 0.05
    xsense_response_contact_gate_gamma: float = 1.0
    xsense_response_contact_gate_blur_passes: int = 1
    xsense_response_contact_rgb: tuple[float, float, float] = (-0.052, -0.002, 0.072)
    xsense_response_edge_rgb: tuple[float, float, float] = (-0.010, 0.0, 0.016)
    xsense_response_edge_gain: float = 1.0
    xsense_response_indent_gamma: float = 0.85
    xsense_response_indent_support: float = 0.35
    xsense_response_taxim_residual_mix: float = 0.0

    device: str = "cuda"

    with_shadow: bool = False

    tactile_img_res: tuple = (320, 240)
    """Resolution of the Tactile Image.

    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is up/down sampled.
    """

    gelpad_height: float = MISSING
    """Used for computing indentation depth from height map"""

    # Asset Data
    gelpad_to_camera_min_distance: float = MISSING
    """Min distance of camera to the gelpad.
    Used for computing the indentation depth out of the
    camera height map.
    """
