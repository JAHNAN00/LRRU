# LRRU Agent Notes

## 工作语言
- 之后在本仓库内默认使用中文与用户沟通。

## 仓库性质
- 这是单仓库 PyTorch 深度补全项目，不是 monorepo，也没有现成的 lint/typecheck/test/CI 配置可复用。
- 训练入口是 `train_apex.py`，验证/推理入口是 `val.py`；主模型通过配置项 `model: 'model_dcnv2'` 动态加载 `model/model_dcnv2.py`。
- 数据集只实现了 KITTI Depth Completion 流程，数据读取在 `dataloaders/kitti_loader.py`，路径规则在 `dataloaders/paths_and_transform.py`。

## 运行命令
- 从仓库根目录执行时，用 `python train_apex.py -c train_lrru_mini_kitti.yml` 和 `python val.py -c val_lrru_mini_kitti.yml`。
- `README.md` 与 `train.sh`/`val.sh` 里的 `python LRRU/...` 写法假设你站在上级目录；在当前仓库根目录直接照抄会错。
- `configs/__init__.py` 会固定执行 `cfg.merge_from_file('configs/' + arg.configuration)`，所以 `-c` 传文件名最稳妥；若传 `configs/...` 会变成重复路径。
- 根目录 `train.sh`、`val.sh` 只是示例快捷脚本，默认跑 mini 配置。

## 依赖与环境坑
- 训练代码硬依赖 NVIDIA Apex：`train_apex.py` 顶层直接 `from apex.parallel import DistributedDataParallel as DDP` 和 `from apex import amp`，没装 Apex 连 import 都过不了。
- 模型硬依赖自定义 DCNv2 扩展；先在 `model/DCNv2` 下执行 `python setup.py build develop`，等价于 `./make.sh`。
- `model/DCNv2/setup.py` 仅在 `torch.cuda.is_available()` 且检测到 `CUDA_HOME` 时编译 CUDA 扩展；没有 CUDA 时不会自动报错，但主模型实际仍依赖该扩展。
- 训练默认会在 rank 0 执行 `wandb.login()`；`record_by_wandb_online: False` 只会设置 `WANDB_MODE=dryrun`，不会移除对 `wandb` 包和登录流程的依赖。
- README 声明的参考环境是 `torch 1.7.1`、`torchvision 0.8.0`、CUDA 12.0、cuDNN 8.5.0；改动底层算子或编译相关代码前先注意兼容性。

## 配置与数据流
- 所有运行参数来自 `configs/config.py` 默认值 + `configs/*.yml` 覆盖；不要假设命令行还能单独覆盖任意字段。
- YAML 里的 `data_folder` 当前是占位路径 `/home/temp_user/kitti_depth`，本地运行前通常必须先改配置。
- 支持的 split 只有 `train`、`val`、`test_completion`、`test_prediction`；其中 `val` 还受 `args.val` 的 `full/select` 分支控制。
- `KittiDepth.__getitem__` 会额外生成 `dep_clear` 和 `ip`，它们不是磁盘字段，而是运行时由稀疏深度经过去异常和 IPBasic 风格补全得到。

## 验证与输出
- `val.py` 默认单卡 `DataParallel` 推理，`test_option` 从 YAML 读取；`val_lrru_mini_kitti.yml` 当前默认是 `test_option: 'val'`、`tta: False`。
- 验证结果会写到 `args.test_dir/test/result_metric.txt`；若开启 `save_test_image`，输出目录也挂在 `args.test_dir/test/` 下。
- 训练会把源码备份到 `save_dir/backup_code`，并在 `save_dir/train`、`save_dir/val` 写日志与可视化产物；排查结果时优先看这些目录。

## 改动建议
- 改训练/验证逻辑时，优先做能单独 import 或单卡运行的最小验证；仓库没有现成自动化测试兜底。
- 若只想验证配置或数据路径，优先检查 `configs/*.yml` 与 `dataloaders/paths_and_transform.py`，不要先从大模型内部下手。
