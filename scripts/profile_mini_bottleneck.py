import argparse
import os
import random
import statistics
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs import get as get_cfg
from dataloaders.NNfill import fill_in_fast
from dataloaders.kitti_loader import KittiDepth
from dataloaders.utils import outlier_removal
from loss import get as get_loss
from model import get as get_model
from optimizer_scheduler import make_optimizer_scheduler


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def summarize_ms(name, values):
    values_ms = [v * 1000.0 for v in values]
    return {
        "name": name,
        "mean_ms": statistics.mean(values_ms),
        "median_ms": statistics.median(values_ms),
        "p90_ms": percentile(values_ms, 0.9),
    }


def print_summary(summary):
    print(
        f"{summary['name']:<20} mean={summary['mean_ms']:.2f} ms | "
        f"median={summary['median_ms']:.2f} ms | p90={summary['p90_ms']:.2f} ms"
    )


def make_cfg(config_name):
    arg = argparse.Namespace(project_name="LRRU", configuration=config_name)
    cfg = get_cfg(arg)
    cfg.defrost()
    cfg.debug_dp = False
    cfg.test = False
    cfg.record_by_wandb_online = False
    cfg.test_record_by_wandb_online = False
    cfg.freeze()
    return cfg


def benchmark_sample_breakdown(dataset, indices):
    print("\n[1] Sample breakdown")
    raw_times = []
    transform_times = []
    outlier_times = []
    ipfill_times = []
    total_times = []

    for idx in indices:
        t0 = time.perf_counter()
        dep, gt, K, rgb, _ = dataset.__getraw__(idx)
        t1 = time.perf_counter()
        dep, gt, K, rgb = dataset.transforms(dataset.split, dataset.args, dep, gt, K, rgb)
        t2 = time.perf_counter()
        dep_np = dep.numpy().squeeze(0)
        dep_clear, _ = outlier_removal(dep_np)
        t3 = time.perf_counter()
        dep_np_ip = np.copy(dep_np)
        _ = fill_in_fast(dep_np_ip, max_depth=100.0, extrapolate=True, blur_type="gaussian")
        t4 = time.perf_counter()

        raw_times.append(t1 - t0)
        transform_times.append(t2 - t1)
        outlier_times.append(t3 - t2)
        ipfill_times.append(t4 - t3)
        total_times.append(t4 - t0)

    for summary in [
        summarize_ms("read_raw", raw_times),
        summarize_ms("transform", transform_times),
        summarize_ms("outlier", outlier_times),
        summarize_ms("ipfill", ipfill_times),
        summarize_ms("getitem_total", total_times),
    ]:
        print_summary(summary)


def benchmark_loader(dataset, batch_size, workers, steps, warmup=3):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    iterator = iter(loader)
    total_samples = 0
    times = []

    for step in range(steps + warmup):
        t0 = time.perf_counter()
        batch = next(iterator)
        t1 = time.perf_counter()
        if step >= warmup:
            times.append(t1 - t0)
            total_samples += batch["gt"].shape[0]

    elapsed = sum(times)
    return {
        "workers": workers,
        "mean_batch_s": statistics.mean(times),
        "samples_per_s": total_samples / elapsed,
    }


def move_batch_to_cuda(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}


def build_train_components(cfg, device):
    model_ctor = get_model(cfg)
    net = model_ctor(cfg).to(device)
    net.train()
    optimizer, _ = make_optimizer_scheduler(cfg, net)
    loss_ctor = get_loss(cfg)
    loss_fn = loss_ctor(cfg, cfg.loss)
    loss_fn.cuda()
    return net, optimizer, loss_fn


def benchmark_compute_only(cfg, batch, steps, use_amp):
    print("\n[3] GPU compute only")
    device = torch.device("cuda:0")
    net, optimizer, loss_fn = build_train_components(cfg, device)
    scaler = GradScaler(enabled=use_amp)
    batch = move_batch_to_cuda(batch, device)
    torch.backends.cudnn.benchmark = True

    times = []
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(steps + 3):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        amp_ctx = autocast(enabled=use_amp) if use_amp else nullcontext()
        with amp_ctx:
            output = net(batch)
            loss_sum, _ = loss_fn(output["results"], batch["gt"])
        if use_amp:
            scaler.scale(loss_sum).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_sum.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        if step >= 3:
            times.append(t1 - t0)

    batch_size = batch["gt"].shape[0]
    mean_step = statistics.mean(times)
    print(
        f"compute_only amp={use_amp} | mean_step={mean_step:.3f}s | "
        f"samples/s={batch_size / mean_step:.2f} | "
        f"peak_mem={torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB"
    )


def benchmark_end_to_end(cfg, dataset, batch_size, workers, steps, use_amp):
    print("\n[4] End-to-end train loop")
    device = torch.device("cuda:0")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    net, optimizer, loss_fn = build_train_components(cfg, device)
    scaler = GradScaler(enabled=use_amp)

    iterator = iter(loader)
    total_samples = 0
    total_elapsed = 0.0
    step_times = []
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(steps + 2):
        t0 = time.perf_counter()
        batch = next(iterator)
        batch = move_batch_to_cuda(batch, device)
        optimizer.zero_grad(set_to_none=True)
        amp_ctx = autocast(enabled=use_amp) if use_amp else nullcontext()
        with amp_ctx:
            output = net(batch)
            loss_sum, _ = loss_fn(output["results"], batch["gt"])
        if use_amp:
            scaler.scale(loss_sum).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_sum.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        if step >= 2:
            step_times.append(t1 - t0)
            total_elapsed += t1 - t0
            total_samples += batch["gt"].shape[0]

    mean_step = statistics.mean(step_times)
    print(
        f"end_to_end workers={workers} amp={use_amp} | mean_step={mean_step:.3f}s | "
        f"samples/s={total_samples / total_elapsed:.2f} | "
        f"peak_mem={torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_lrru_mini_auto_kitti.yml")
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--loader-steps", type=int, default=24)
    parser.add_argument("--train-steps", type=int, default=12)
    parser.add_argument("--workers", nargs="*", type=int, default=[0, 1, 2, 3, 4, 6])
    parser.add_argument("--seed", type=int, default=1128)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = make_cfg(args.config)
    dataset = KittiDepth("train", cfg)
    indices = random.sample(range(len(dataset)), args.sample_count)

    print(f"Config: {args.config}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Num threads in config: {cfg.num_threads}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    benchmark_sample_breakdown(dataset, indices)

    print("\n[2] DataLoader throughput")
    for workers in args.workers:
        result = benchmark_loader(dataset, cfg.batch_size, workers, args.loader_steps)
        print(
            f"workers={result['workers']} | mean_batch={result['mean_batch_s']:.3f}s | "
            f"samples/s={result['samples_per_s']:.2f}"
        )

    if not torch.cuda.is_available():
        return

    warm_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=True)
    first_batch = next(iter(warm_loader))
    benchmark_compute_only(cfg, first_batch, args.train_steps, use_amp=True)
    benchmark_compute_only(cfg, first_batch, args.train_steps, use_amp=False)
    benchmark_end_to_end(cfg, dataset, cfg.batch_size, workers=min(4, os.cpu_count() or 1), steps=args.train_steps, use_amp=True)


if __name__ == "__main__":
    main()
