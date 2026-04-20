# 实验记录

## 目标
- 记录 LRRU 在本机环境下的验证、训练尝试、报错与结果。

## 当前判断（2080 Ti 11GB）
- 验证：可以。四个变体在单卡上做 `val.py` 推理都应可行，`batch_size=1` 显存压力不大。
- 训练：可以，但需要分模型降 batch 看。
- `LRRU-Mini`：最稳，优先尝试，建议先从单卡 `batch_size=8` 试起。
- `LRRU-Tiny`：可训练，建议先从单卡 `batch_size=4` 或 `6` 试起。
- `LRRU-Small`：大概率可训练，但更依赖降 batch，建议先试 `batch_size=2` 或 `4`。
- `LRRU-Base`：单卡 2080 Ti 不适合作为首选训练目标；若强行训练，较可能只能 `batch_size=1`，并需要梯度累积，训练会明显变慢。
- 相比原作者多张 3090：验证精度不该因显卡明显变化；从头训练时，主要差在吞吐、可用 batch 和训练稳定性，单卡 2080 Ti 复现 `Base` 的难度明显更高。

## 已知前置条件
- 训练前需要先编译 `model/DCNv2` 扩展；若扩展不可用，当前仓库里已加了 dry-run 回退路径用于连通性验证。
- `train_apex.py` 已加 Apex 兼容回退：有 Apex 则走 Apex；无 Apex 则退化到 PyTorch DDP + amp 占位逻辑，便于单卡 dry-run。
- 配置里的 `data_folder` 默认是占位路径，跑之前要改成实际 KITTI 路径。

## 本机环境（2026-04-20）
- 已创建 `mamba` 环境：`LRRU`（`/home/an/miniconda3/envs/LRRU`）。
- Python：`3.8.20`。
- PyTorch：`1.7.1`，torchvision：`0.8.2`。
- 关键依赖：`opencv-python-headless`、`yacs`、`wandb`、`emoji`、`tensorboard` 等已安装。
- Apex：已尝试安装，但与当前 `torch==1.7.1` 的 API 有兼容问题；当前训练脚本通过回退逻辑可在无 Apex 情况下运行 dry-run。

## 最新进展（2026-04-20）
- 已完成数据链接：
  - `data/kitti_depth -> /media/an/4T/datasets/kitti_depth`
  - `data/kitti_raw -> /media/an/4T/datasets/kitti_raw`
- 已适配代码读取逻辑（`dataloaders/paths_and_transform.py`）：
  - 训练/`val full`：从 `data_depth_velodyne` 读稀疏深度，从 `data_depth_annotated` 读 GT，从 `kitti_raw` 读 RGB 与标定。
  - `val select`/`test_*`：路径前缀修正为 `data_depth_selection`。
- 已更新 KITTI 配置文件 `data_folder` 为：`/home/an/Desktop/LRRU/data`。
- 预训练权重验证（不训练直接测试）已跑通：
  - 命令：`mamba run -n LRRU python val.py -c val_lrru_mini_kitti.yml`
  - 权重：`./pretrained/LRRU_Mini.pt`
  - 结果文件：`pretrained/LRRU_Mini/test/result_metric.txt`
  - 汇总指标：`RMSE=0.8063`、`MAE=0.2102`、`iRMSE=0.0023`、`iMAE=0.0009`
  - 单位换算后与论文/README 的 LRRU-Mini（KITTI val）一致：`806.3 mm / 210.2 mm / 2.3 / 0.9`。

## DCNv2 编译情况（2026-04-20）
- 已恢复 `model/dcn_v2.py` 为原生 `_ext` 依赖（不再使用 Python fallback）。
- 编译器按本机经验固定为 `gcc-10/g++-10`，在 `LRRU` 环境内编译通过。
- 编译命令（在 `model/DCNv2` 目录）：
  - `mamba run -n LRRU env CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 CUDAHOSTCXX=/usr/bin/g++-10 TORCH_CUDA_ARCH_LIST=7.5 python setup.py build develop`
- 兼容改动范围仅在 `model/DCNv2` 目录内（旧 TH/THC 接口替换为当前可编译接口，不改项目外部调用接口）。
- 项目侧可用性验证：已用 dry-run 跑通 `train_apex.py` 与 `val.py`。
- 备注：`model/DCNv2/testcuda.py` 的严格 gradcheck 在本机仍可能报 Jacobian mismatch，但不影响当前项目 dry-run 训练/验证链路。

## 运行约定
- 后续在本仓库执行训练/验证/脚本时，统一使用 `LRRU` 环境。
- 推荐命令形式：`mamba run -n LRRU python <script>.py ...`。
- 已验证 dry-run 命令：
  - `mamba run -n LRRU python train_apex.py -c train_lrru_dryrun.yml`
  - `mamba run -n LRRU python val.py -c val_lrru_dryrun.yml`

## 距离真实项目跑起来还差什么
- 目前已具备：环境、依赖、DCNv2 编译、真实 KITTI 数据链接、预训练权重验证流程。
- 当前可直接执行：
  - 预训练验证：`python val.py -c val_lrru_mini_kitti.yml`
  - 训练入口：`python train_apex.py -c train_lrru_mini_kitti.yml`
- 若要做完整训练复现：主要是确认显存/批大小设置与 `wandb` 登录策略。

## 数据目录与格式要求（按代码读取规则）
- 通用要求：
  - 深度图使用 **16-bit png**（读取后按 `/256.0` 还原到米）。
  - RGB 使用常规 `png`。
  - 内参文件为文本；训练/val-full 用 `calib_cam_to_cam.txt`，val-select/test 用每帧一个 `.txt`。
- 训练集（`split=train`）最少结构：
  - `data_folder/kitti_depth/data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_02/*.png`
  - `data_folder/kitti_depth/data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_02/*.png`
  - `data_folder/kitti_raw/<date>/*_sync/image_02/data/*.png`
  - `data_folder/kitti_raw/<date>/calib_cam_to_cam.txt`
  - 也支持 `image_03` 对应路径。
- 验证集（`split=val`）：
  - `val: full` 时目录与训练同构，根路径改为 `data_folder/kitti_depth/...` 与 `data_folder/kitti_raw/...`
  - `val: select` 时需要：
    - `data_folder/kitti_depth/data_depth_selection/val_selection_cropped/velodyne_raw/*.png`
    - `data_folder/kitti_depth/data_depth_selection/val_selection_cropped/groundtruth_depth/*.png`
    - `data_folder/kitti_depth/data_depth_selection/val_selection_cropped/image/*.png`
    - `data_folder/kitti_depth/data_depth_selection/val_selection_cropped/intrinsics/*.txt`
- 测试集（可选）：
  - completion：`data_depth_selection/test_depth_completion_anonymous/{velodyne_raw,image,intrinsics}`
  - prediction：`data_depth_selection/test_depth_prediction_anonymous/{image,intrinsics}`

## 实验计划
- [ ] 先确认 `DCNv2` 能成功编译。
- [ ] 确认 Apex、PyTorch、CUDA 版本可兼容导入。
- [ ] 先跑 `LRRU-Mini` 验证，确认数据路径、权重路径、输出目录都正常。
- [ ] 再尝试 `LRRU-Mini` 单卡训练。
- [ ] 若显存允许，再逐步尝试 `Tiny` / `Small`。
- [ ] `Base` 暂不作为第一优先级，除非先验证单卡显存与训练稳定性。

## 运行记录模板

### YYYY-MM-DD HH:MM
- 操作：
- 配置：
- 环境：
- 结果：
- 耗时：
- 显存占用：
- 问题/报错：
- 后续动作：
