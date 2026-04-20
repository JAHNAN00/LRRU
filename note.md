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

## 运行约定
- 后续在本仓库执行训练/验证/脚本时，统一使用 `LRRU` 环境。
- 推荐命令形式：`mamba run -n LRRU python <script>.py ...`。
- 已验证 dry-run 命令：
  - `mamba run -n LRRU python train_apex.py -c train_lrru_dryrun.yml`
  - `mamba run -n LRRU python val.py -c val_lrru_dryrun.yml`

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
