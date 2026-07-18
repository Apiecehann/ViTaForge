# Wrist Camera 与 Panda Hand 绑定参数台账

更新时间：2026-07-07

这个文件用来记录 `insert_USB` 当前 wrist camera 与 Franka `panda_hand` 的绑定关系、原始参数、以及已经修改后的参数。后续如果要继续微调视角，可以直接在本文档里改目标值，然后让 Codex 按这里同步修改 USD。

## 生效文件

当前 `insert_USB.py` 读取的 wrist camera prim 是：

```text
/World/envs/env_.*/Robot/WristCamera/Camera
```

它来自这个机器人组合 USD：

```text
third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist.usd
```

这个组合 USD 会 payload 独立相机资产：

```text
third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/WristCamera.usd
```

实际要改当前仿真里的 wrist 视角，优先改组合 USD 里的：

```text
/panda/WristCamera/Camera
```

因为组合 USD 会覆盖 `WristCamera.usd` 里的部分原始相机参数。

## 备份文件

完全原始备份：

```text
third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist_original.usd
```

已经改过广角、但还没有改 `orient` 的备份：

```text
third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist_before_orient_15deg.usd
```

上一版不满意的复合 `orient` 备份：

```text
third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist_before_z_only_15deg.usd
```

只绕 Camera local `+Z` 旋转 15 度的 roll 测试备份：

```text
third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist_before_pitch_down_15deg.usd
```

## 原始参数

来源：

```text
uipc_gelpads_high_res_wrist_original.usd
```

### 相机外壳根节点

prim：

```text
/panda/WristCamera
```

参数：

```text
xformOp:translate = (0.20817375707697683, 0.053511678010775304, 0.8385946361287533)
xformOp:rotateXYZ = (0, 0, 159.35109)
xformOp:scale = (1, 1, 1)
xformOpOrder = [xformOp:translate, xformOp:rotateXYZ, xformOp:scale]
```

说明：这一层控制整个 wrist camera 外壳/根节点相对于 `/panda` 的装配姿态。

### 真正拍图的 Camera prim

prim：

```text
/panda/WristCamera/Camera
```

参数：

```text
xformOp:translate = (0.07644610010979583, 0.05320889805378959, 0.02467079308763187)
xformOp:orient = (0.21643962, 0, 0, 0.976296)
xformOp:scale = (1, 1, 1)
xformOpOrder = [xformOp:translate, xformOp:orient, xformOp:scale]

focalLength = 1.940000057220459
horizontalAperture = 2.687999963760376
verticalAperture = 15.290800094604492
clippingRange = (0.01, 100)
focusDistance = 0.0
projection = perspective
```

说明：

```text
xformOp:translate  控制 Camera 光心相对于 /panda/WristCamera 的位置
xformOp:orient     控制 Camera 光轴方向和画面 roll，四元数顺序是 (w, x, y, z)
focalLength        越小越广角
horizontalAperture 越大越广角
clippingRange      深度/渲染裁剪范围
```

### 与 panda_hand 的固定关节

prim：

```text
/panda/panda_hand/wrist_camera
```

参数：

```text
physics:body0 = ['/panda/panda_hand']
physics:body1 = ['/panda/WristCamera']
proxyPrim = []

physics:localPos0 = (6.1747324e-10, -8.808347e-10, -1.110223e-16)
physics:localRot0 = (1, -3.51006e-18, 3.276206e-18, -6.4181377e-10)
physics:localPos1 = (0.09358327, 0.09245222, 0.08740536)
physics:localRot1 = (-1.451128e-17, -0.54206693, 0.8403353, 6.0424075e-17)

physics:jointEnabled = True
physics:excludeFromArticulation = False
physics:collisionEnabled = False
```

说明：这说明 `/panda/WristCamera` 是通过 `PhysicsFixedJoint` 固定到 `/panda/panda_hand` 上的。也就是说 wrist camera 跟着 `panda_hand` 走，不是直接绑在 finger 或 gelpad 上。

### 原始合成结果

把 `/panda/WristCamera`、`/panda/WristCamera/Camera` 和 fixed joint 合成以后，Camera 相对于 `panda_hand` 的结果是：

```text
Camera relative to panda_hand pos = (0.0428181985, -0.000568497696, 0.0627345698)
Camera relative to panda_hand quat(wxyz) = (-1.60812265e-16, 0.703091231, 0.711099656, 2.37525776e-17)

Camera optical local -Z in panda_hand = (4.68652067e-17, -1.1314261e-16, 1)
Camera up local +Y in panda_hand = (0.999935865, 0.0113254411, -4.5580811e-17)
Camera right local +X in panda_hand = (-0.0113254411, 0.999935865, 1.13666123e-16)
```

直观理解：原始光轴基本朝 `panda_hand +Z`，也就是大致朝 gripper/指尖前方。

## 当前已改动后的参数

来源：

```text
uipc_gelpads_high_res_wrist.usd
```

### 相机外壳根节点

prim：

```text
/panda/WristCamera
```

当前未改动，仍为：

```text
xformOp:translate = (0.20817375707697683, 0.053511678010775304, 0.8385946361287533)
xformOp:rotateXYZ = (0, 0, 159.35109)
xformOp:scale = (1, 1, 1)
xformOpOrder = [xformOp:translate, xformOp:rotateXYZ, xformOp:scale]
```

### 真正拍图的 Camera prim

prim：

```text
/panda/WristCamera/Camera
```

当前参数：

```text
xformOp:translate = (0.07644610010979583, 0.05320889805378959, 0.02467079308763187)
xformOp:orient = (0.21458794, -0.028251039, -0.1274322, 0.96794367)
xformOp:scale = (1, 1, 1)
xformOpOrder = [xformOp:translate, xformOp:orient, xformOp:scale]

focalLength = 1.5
horizontalAperture = 3.0
verticalAperture = 15.290800094604492
clippingRange = (0.01, 100)
focusDistance = 0.0
projection = perspective
```

已改动项：

```text
focalLength:        1.940000057220459 -> 1.5
horizontalAperture: 2.687999963760376 -> 3.0
xformOp:orient:     (0.21643962, 0, 0, 0.976296)
                    -> (0.21458794, -0.028251039, -0.1274322, 0.96794367)
```

`orient` 的改动方式：

```text
在原始 orient 基础上，只追加相机局部坐标旋转：
1. 绕 Camera local -X 旋转 15 度
```

### 与 panda_hand 的固定关节

prim：

```text
/panda/panda_hand/wrist_camera
```

当前未改动，仍为：

```text
physics:body0 = ['/panda/panda_hand']
physics:body1 = ['/panda/WristCamera']
proxyPrim = []

physics:localPos0 = (6.1747324e-10, -8.808347e-10, -1.110223e-16)
physics:localRot0 = (1, -3.51006e-18, 3.276206e-18, -6.4181377e-10)
physics:localPos1 = (0.09358327, 0.09245222, 0.08740536)
physics:localRot1 = (-1.451128e-17, -0.54206693, 0.8403353, 6.0424075e-17)

physics:jointEnabled = True
physics:excludeFromArticulation = False
physics:collisionEnabled = False
```

### 当前合成结果

把当前 `/panda/WristCamera`、`/panda/WristCamera/Camera` 和 fixed joint 合成以后，Camera 相对于 `panda_hand` 的结果是：

```text
Camera relative to panda_hand pos = (0.0428181985, -0.000568497696, 0.0627345698)
Camera relative to panda_hand quat(wxyz) = (0.0917718184, 0.697076194, 0.705016095, 0.0928171276)

Camera optical local -Z in panda_hand = (-0.258802438, -0.00293123804, 0.965925828)
Camera up local +Y in panda_hand = (0.965863879, 0.0109395217, 0.258819037)
Camera right local +X in panda_hand = (-0.0113254268, 0.999935865, 2.04783749e-09)
```

直观理解：当前位置没有变，画面左右方向基本没有 roll；光轴从原始的接近 `panda_hand +Z`，向 `panda_hand -X` 方向倾斜约 15 度，用来让画面看到下方更多范围。

## 修改指南

常见目标和应该改的字段：

```text
想让画面更广：
  降低 /panda/WristCamera/Camera 的 focalLength
  或增大 /panda/WristCamera/Camera 的 horizontalAperture

想让画面更窄：
  增大 focalLength
  或降低 horizontalAperture

想让镜头朝向改变：
  改 /panda/WristCamera/Camera 的 xformOp:orient

想让光心位置变：
  改 /panda/WristCamera/Camera 的 xformOp:translate

想让整个相机外壳/安装位姿变：
  改 /panda/WristCamera 的 xformOp:translate / xformOp:rotateXYZ
  或改 /panda/panda_hand/wrist_camera fixed joint 的 localPos/localRot

想改输出图像尺寸：
  改 envs/insert_USB.py 里的 wrist CameraCfg width / height
```

注意：

```text
Camera 默认沿自己的 local -Z 方向看。
Camera local +Y 是图像上方。
Camera local +X 是图像右方。
```

## 最终调整

```text
目标文件:
  third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist.usd

目标 prim:
  /panda/WristCamera/Camera

最终参数:
  xformOp:translate = (0.07644610010979583, 0.05320889805378959, 0.02467079308763187)
  xformOp:orient = (0.21458794, -0.028251039, -0.1274322, 0.96794367)
  focalLength = 1.5
  horizontalAperture = 3.0
  clippingRange = (0.01, 100)

目标 prim:
  /panda/WristCamera

最终参数:
  xformOp:translate = (0.20817375707697683, 0.053511678010775304, 0.8385946361287533)
  xformOp:rotateXYZ = (0, 0, 159.35109)

目标 fixed joint:
  /panda/panda_hand/wrist_camera

最终参数:
  physics:localPos0 = (6.1747324e-10, -8.808347e-10, -1.110223e-16)
  physics:localRot0 = (1, -3.51006e-18, 3.276206e-18, -6.4181377e-10)
  physics:localPos1 = (0.09358327, 0.09245222, 0.08740536)
  physics:localRot1 = (-1.451128e-17, -0.54206693, 0.8403353, 6.0424075e-17)
```

最终改动只落在 `/panda/WristCamera/Camera`：

```text
focalLength:        1.940000057220459 -> 1.5
horizontalAperture: 2.687999963760376 -> 3.0
xformOp:orient:     (0.21643962, 0, 0, 0.976296)
                    -> (0.21458794, -0.028251039, -0.1274322, 0.96794367)
```

最终效果摘要：

```text
Camera relative to panda_hand pos = (0.0428181985, -0.000568497696, 0.0627345698)
Camera relative to panda_hand quat(wxyz) = (0.0917718184, 0.697076194, 0.705016095, 0.0928171276)

Camera optical local -Z in panda_hand = (-0.258802438, -0.00293123804, 0.965925828)
Camera up local +Y in panda_hand = (0.965863879, 0.0109395217, 0.258819037)
Camera right local +X in panda_hand = (-0.0113254268, 0.999935865, 2.04783749e-09)
```

直观结论：相机光心、外壳根节点和 fixed joint 都保持原位；最终只把 wrist camera 变成更广角，并在原始 `orient` 基础上绕 Camera local `-X` 旋转 15 度，让光轴向 `panda_hand -X` 方向倾斜约 15 度。
