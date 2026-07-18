### 路径
`envs/robot/robot_cfg.py`

```python
def create_franka_gsmini_gripper(data_type:list[str]):
    robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # "panda_joint1": 0.0,
                # "panda_joint2": 0.0,
                # "panda_joint3": 0.0,
                # "panda_joint4": -2.46,
                # "panda_joint5": 0.0,
                # "panda_joint6": 2.5,
                # "panda_joint7": 0.741,
                "panda_joint1": -0.010809095,
                "panda_joint2": 0.096037410,
                "panda_joint3": 0.000734462,
                "panda_joint4": -2.433035851,
                "panda_joint5": 0.035354517,
                "panda_joint6": 2.500859022,
                # "panda_joint7": 1.543236494,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ),
    )
```