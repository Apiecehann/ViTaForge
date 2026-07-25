增加了保存语言指令的功能：

[scripts/collect_data.py (line 106)]：新增 save_instruction 开关逻辑；开启后读取任务文件里的 TASK_INSTRUCTION。

[scripts/collect_data.py (line 118)]：每次 reset 后把固定指令写入 task。

[scripts/parallel_collect_data.py (line 40)]：并行数采也支持同样逻辑。

[envs/_base_task.py (line 659)]：成功保存 HDF5 后写入 root attr：f.attrs["instruction"]。

[envs/_base_task.py (line 666)]：set_instruction() 同时写入 metadata["instruction"]。

[task_config/demo.yml (line 24)]：新增默认关闭开关。