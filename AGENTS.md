# Repository Guidelines

## 项目结构与模块组织
`envs/` 存放任务环境、脚本式数据采集策略、机器人封装以及传感器和工具代码。`scripts/` 是采集、回放、评测、可视化等 Python 入口。`policy/` 包含基线策略实现，例如 `ACT/`、`Ablation/`、`ViTAL/`，各自维护训练、部署和配置文件。`encoder/` 是触觉编码器训练代码。`task_config/` 存放任务 YAML 配置，`assets/` 存放 USD 场景、物体和纹理，`docs/` 存放安装与使用文档。`third_party/TacEx/` 视为 vendored code，除非是 UniVTAC 集成问题，否则不要随意修改。

## 构建、测试与开发命令
先创建环境：`conda create -n UniVTAC python=3.10`，再根据 [requirements.txt](/data2/home/liqin/VLA/UniVTAC/requirements.txt) 安装依赖。常用入口命令如下：

- `bash collect_data.sh lift_bottle demo 0`：单进程采集数据。
- `bash parallel_collect.sh lift_bottle demo 0 3`：并行启动多个采集 worker。
- `bash eval_policy.sh lift_bottle demo ACT/deploy.yml 0`：评测一个策略配置或检查点。
- `bash parallel_eval.sh lift_bottle demo ACT/deploy.yml 0 2 100`：并行评测多轮 episode。
- `python encoder/train.py ...`：直接训练触觉编码器。

## 代码风格与命名约定
Python 代码使用 4 个空格缩进，优先遵循相邻文件的既有风格。模块、函数、YAML 文件和 shell 脚本统一使用 `snake_case`。任务名应与模块名保持一致，例如 `envs/lift_bottle.py` 对应 `task_config/demo.yml` 中的任务配置。优先做小范围、任务导向的修改，不做无关的大重构。仓库根目录没有统一 formatter，因此导入顺序、参数命名和日志风格应与周边代码保持一致。

## 测试指南
仓库根目录目前没有 first-party 自动化测试套件。修改后应做最小且有效的验证：采集逻辑优先运行 `collect_data.sh` 或 `parallel_collect.sh`，评测逻辑优先运行 `eval_policy.sh`，编码器相关改动优先运行针对性的 `python encoder/train.py`。如果确实修改了 vendored `TacEx` 代码，还需要在 `third_party/TacEx/` 范围内补跑对应 `pytest`，并在提交说明中明确写出额外影响范围。

## Agent 工作规则
默认先阅读相关代码和文档，再开始修改。修改前先说明预计影响范围，并把改动限制在和当前任务直接相关的最小文件集合。提交前必须运行最小必要验证。面向协作者的说明使用中文，代码、命令和文件名保持英文。不要修改无关文件。

## 实验原则
实验必须服务于明确假设或具体决策。优先做能支持判断的定向实验，不要为了补齐表格而穷举低价值 ablation，也不要做不会改变工程或研究决策的无效 sweep。

## 提交与 Pull Request 约定
近期提交历史以简短祈使句为主，例如 `fix curobo planner`、`update process_data.py`。提交信息应聚焦单一主题，必要时附带 issue 编号，例如 `remove epoch from training config (issue #7)`。Pull Request 应说明影响的任务或策略、依赖的配置或数据集，并附上验证证据。如果改动影响渲染画面、触觉输出或评测产物，应补充截图、视频路径或结果目录说明。
