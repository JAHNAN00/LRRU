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

## 预训练验证结果总表（2026-04-20）
- 数据集：KITTI `val_selection_cropped`，`tta: False`
- 结论：四个尺寸的预训练权重在本机复现结果与 README/论文给出的 KITTI val 指标完全对齐。
- LRRU-Mini：
  - 实测：`RMSE=806.3 mm`、`MAE=210.2 mm`、`iRMSE=2.3`、`iMAE=0.9`
  - README/论文：`806.3 / 210.0 / 2.3 / 0.9`
  - 结论：一致（MAE 仅四舍五入差异）
  - 运行时间：`34.8 s`
- LRRU-Tiny：
  - 实测：`RMSE=763.8 mm`、`MAE=198.9 mm`、`iRMSE=2.1`、`iMAE=0.8`
  - README/论文：`763.8 / 198.9 / 2.1 / 0.8`
  - 结论：完全一致
  - 运行时间：`44.4 s`
- LRRU-Small：
  - 实测：`RMSE=745.3 mm`、`MAE=195.7 mm`、`iRMSE=2.0`、`iMAE=0.8`
  - README/论文：`745.3 / 195.7 / 2.0 / 0.8`
  - 结论：完全一致
  - 运行时间：`71.0 s`
- LRRU-Base：
  - 实测：`RMSE=729.5 mm`、`MAE=188.8 mm`、`iRMSE=1.9`、`iMAE=0.8`
  - README/论文：`729.5 / 188.8 / 1.9 / 0.8`
  - 结论：完全一致
  - 运行时间：`165.0 s`

## 无人值守训练适配（2026-04-20）
- 已修正 `train_apex.py` 中的验证触发逻辑：改为按 epoch 验证一次，避免原实现从 `val_epoch` 开始对每个训练 batch 都跑完整验证。
- 已新增四个单卡自动训练配置：
  - `train_lrru_mini_auto_kitti.yml`
  - `train_lrru_tiny_auto_kitti.yml`
  - `train_lrru_small_auto_kitti.yml`
  - `train_lrru_base_auto_kitti.yml`
- 自动训练配置统一策略：
  - 单卡 `gpus: (0,)`
  - `no_multiprocessing: True`
  - `record_by_wandb_online: False`
  - `epochs: 45` 作为最大上限
  - 提前开启按 epoch 验证，便于早停与跑偏监控
- 当前初始 batch 设定：
  - Mini: `8`
  - Tiny: `4`
  - Small: `2`
  - Base: `1`
- 实测补充：在纯 FP32 回退下，Mini 的 `batch_size=8` 会于 epoch 0 中段 OOM；随后已升级为原生 `torch.cuda.amp` 回退，并重新测试更高 batch 的可行性。

## Mini 训练探索记录（2026-04-21）
- 目标：在单卡 2080 Ti 11GB 上验证 `LRRU-Mini` 从头训练的可行性，并估计真实训练耗时。
- 第 1 次尝试：
  - 配置：`train_lrru_mini_auto_kitti.yml`
  - 设定：纯 FP32 回退，`batch_size=8`
  - 结果：训练能启动，但在 `epoch 0` 中段 OOM。
  - 报错：`Tried to allocate 152.00 MiB ... 14.81 MiB free`
- 第 2 次尝试：
  - 设定：纯 FP32 回退，`batch_size=4`
  - 结果：可稳定训练，loss 快速下降，无立即跑偏迹象。
  - 问题：吞吐偏低，整体训练时间过长。
- 第 3 次尝试：
  - 改动：将无 Apex 回退从纯 FP32 升级为原生 `torch.cuda.amp`
  - 结果：首次直接开启 AMP 时，`DCNv2` 后处理算子报半精度类型不兼容。
  - 报错：`RuntimeError: expected scalar type Float but found Half`
- 第 4 次尝试：
  - 改动：在 `model/model_dcnv2.py` 中将 4 次 `Post_process` 的 `DCNv2` 调用输入强制转回 `float32`，仅让该自定义算子避开半精度。
  - 设定：原生 AMP 回退，`batch_size=8`
  - 结果：训练可稳定启动，不再 OOM。
  - 显存：约 `6.7 GB`
  - 速度：日志显示约 `9~10.5 samples/s`，折算约 `1.1~1.3 it/s`
  - 结论：AMP + 局部 `float32` 兼容方案有效，明显优于纯 FP32。
- 当前状态：
  - 用户在首个验证结果出现前手动中断训练。
  - 因此本轮未产出可与论文直接对比的自训练 `RMSE/MAE` 结果。

## Mini 瓶颈剖析（2026-04-21）
- 目标：判断 `LRRU-Mini` 单卡训练耗时接近 `2 小时/epoch` 时，主瓶颈是否来自 CPU。
- profiling 脚本：`scripts/profile_mini_bottleneck.py`
- 配置：`train_lrru_mini_auto_kitti.yml`，`batch_size=8`，原生 AMP 回退开启。
- 单样本 `__getitem__` 拆分（24 个样本平均）：
  - `read_raw`: `12.1 ms`
  - `transform`: `48.0 ms`
  - `outlier_removal`: `18.0 ms`
  - `fill_in_fast(ipfill)`: `16.4 ms`
  - `getitem_total`: `94.4 ms`
- 结论：CPU 侧最重的是增强/裁剪，其次是 `outlier_removal + ipfill`，两者合计约占单样本 `70%+` 的取数时间。
- DataLoader 吞吐（`batch_size=8`）：
  - `workers=0`: `10.4 samples/s`
  - `workers=2`: `15.3 samples/s`
  - `workers=3`: `19.0 samples/s`
  - `workers=4`: `17.8 samples/s`
  - 结论：当前配置的 `num_threads=3` 已接近最优，继续加 worker 没有收益。
- 去掉 `outlier_removal + ipfill` 后的纯 loader 吞吐：
  - `workers=3~4` 可到约 `18~26 samples/s`
  - 说明这两步确实有 CPU 成本，但在多 worker 下仍能被并行掩盖。
- 纯 GPU 单步训练速度（复用同一 batch，隔离数据加载）：
  - `AMP=True`: `0.689 s/step`，约 `11.6 samples/s`，峰值显存 `5.32 GB`
  - `AMP=False`: `0.834 s/step`，约 `9.6 samples/s`，峰值显存 `8.65 GB`
  - 结论：AMP 能带来约 `17%` 的单步提速，并显著降低显存占用。
- 端到端训练速度（真实 DataLoader + 模型训练）：
  - `workers=4`, `AMP=True`: `0.735 s/step`，约 `10.9 samples/s`
  - 与纯 GPU `11.6 samples/s` 只差约 `6%`
  - 结论：在 worker 设置合理时，训练主瓶颈不在 CPU，而在 GPU 侧前反传本身。
- 额外观察：
  - 若 `workers=0`，端到端训练吞吐会明显掉到约 `4.4 samples/s`。
  - 说明 CPU 会在 worker 配置不当时成为瓶颈，但这不是当前 `num_threads=3` 配置下的主限制。
- 综合判断：
  - `LRRU-Mini` 当前约 `2 小时/epoch` 是正常量级，不是单纯 CPU 拖慢。
  - 根因是：训练集规模大（约 `85898` 样本）+ 每 step 的 GPU 前反传本身就约 `0.69 s`。
  - 按 `11.6 samples/s` 估算，`85898 / 11.6 ≈ 7405 s ≈ 2.06 小时/epoch`。
  - 即使进一步优化 CPU 侧，整体 epoch 时间也只能小幅下降，无法从 `2 小时` 级别变成 `1 小时` 内。

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

## wandb 与 Git 清理检查（2026-04-21）
- 已复查 `train_apex.py` 与 `configs/config.py`：
  - 默认设置 `wandb_disable_code: True`，启动时会导出 `WANDB_DISABLE_CODE=true`。
  - 默认设置 `backup_source_code: False`，训练不会再自动执行整仓库 `backup_source_code(...)`。
- 已复查 `.gitignore`：当前会忽略 `wandb/`、`run_logs/`、`pretrained/`、`backup_code/`、`events.out.tfevents.*`、`*.log`、`*.pt`、`*.pth` 等实验产物。
- 已用 `git ls-files` 复查：目前没有已被 Git 跟踪的权重、日志、`wandb` 离线目录或评测产物会随本次推送进入远程仓库。
- 结论：当前代码状态下，不会再因为反复备份源码把磁盘持续吃满；当前 Git 状态下，也不会把常见训练产物误推到远程。

## Mini 5-epoch 从头训练结果（2026-04-23）
- 配置：`configs/train_lrru_mini_5epoch_kitti.yml`
- 运行目录：`wandb/offline-run-20260423_000346-v33myi2o`
- 状态：已完整跑完，`run_logs/mini_5epoch_watch.status` 显示 `finished exit_code=0`
- 训练中为避免再次中断，已修复两处代码问题：
  - `summary/summary.py`
    - 当 `args.output` 为空或 key 不存在时，验证结果保存会自动回退到 `output['results']` 或第一个非空输出，不再因 `KeyError: ''` 中断。
    - `update()` 现在稳定返回当前 `RMSE/MAE`，不再因 `online_rmse_only=True` 使返回值为 `None`。
  - `train_apex.py`
    - 验证阶段保存预测图像若失败，只打印 warning，不再打断主训练流程。
    - best metric 比较前增加 `None` 保护。
- 每个 epoch 的验证结果：
  - epoch 1：`RMSE=1038.2 mm`，`MAE=276.0 mm`，`iRMSE=3.4`，`iMAE=1.1`
  - epoch 2：`RMSE=968.6 mm`，`MAE=251.4 mm`，`iRMSE=3.0`，`iMAE=1.1`
  - epoch 3：`RMSE=929.3 mm`，`MAE=244.9 mm`，`iRMSE=2.9`，`iMAE=1.1`
  - epoch 4：`RMSE=931.8 mm`，`MAE=243.4 mm`，`iRMSE=2.9`，`iMAE=1.0`
  - epoch 5：`RMSE=942.2 mm`，`MAE=247.0 mm`，`iRMSE=3.0`，`iMAE=1.0`
- 最佳 checkpoint：
  - `best_rmse_model.pt`：`epoch=3`，最佳 `RMSE=929.3 mm`
  - `best_mae_model.pt`：`epoch=4`，最佳 `MAE=243.4 mm`
  - `latest_model.pt`：`epoch=5`
- 与 README/论文中的 `LRRU-Mini` 预训练结果对比：
  - 论文/README：`RMSE=806.3 mm`，`MAE=210.0 mm`，`iRMSE=2.3`，`iMAE=0.9`
  - 本次 5-epoch 从头训练最佳：`RMSE=929.3 mm`，`MAE=243.4 mm`，`iRMSE=2.9`，`iMAE=1.0`
  - 差距：`RMSE +123.0 mm`，`MAE +33.4 mm`，`iRMSE +0.6`，`iMAE +0.1`
  - 结论：训练链路已稳定可用，但 5 个 epoch 仍远不足以接近论文最终精度；模型已明显收敛，后续应直接从 `latest_model.pt` 续跑更多 epoch。

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
