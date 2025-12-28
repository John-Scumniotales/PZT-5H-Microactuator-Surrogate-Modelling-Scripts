"""
Script for running batched distributed inference for surrogates
across multiple GPUs
"""

import argparse
import csv
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_MODEL_ROOT = Path("Model1")
DEFAULT_TOLERANCE = 0.02
DEFAULT_BATCH_SIZE = 128
DEFAULT_TIMING_REPEATS = 1
DEFAULT_WARMUP_BATCHES = 1
DEFAULT_NUM_MODELS: Optional[int] = None
DEFAULT_OUTPUT_CSV = Path("slurm_out/sweep/surrogate_results.csv")
DEFAULT_MAX_WORKERS = 512

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class Seq2SeqForecaster(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, out_dim: int = 3, num_layers: int = 2):
        super().__init__()
        dropout = 0.1 if num_layers > 1 else 0.0
        self.encoder = torch.nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder = torch.nn.GRU(out_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = torch.nn.Linear(hidden_dim, out_dim)
        self.horizon = horizon
        self.out_dim = out_dim

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        batch_size = src.size(0)
        _, h = self.encoder(src)
        dec_in = torch.zeros(batch_size, 1, self.out_dim, device=src.device, dtype=src.dtype)
        outputs = []
        for _ in range(self.horizon):
            dec_out, h = self.decoder(dec_in, h)
            step = self.proj(dec_out.squeeze(1))
            outputs.append(step)
            dec_in = step.unsqueeze(1)
        return torch.stack(outputs, dim=1)


def load_checkpoint(
    model_root: Path,
    device: torch.device,
    input_dim: int,
    horizon: int,
    hidden_dim: int = 64,
) -> Seq2SeqForecaster:
    model_path = model_root / "seq2seq_surrogate.pt"
    best_ckpt_path = model_root / "checkpoints" / "best_seq2seq.pt"
    state: Optional[dict] = None
    config = {"input_dim": input_dim, "hidden_dim": hidden_dim, "horizon": horizon}

    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state", ckpt)
            config.update(ckpt.get("config", {}))

    if state is None and best_ckpt_path.exists():
        state = torch.load(best_ckpt_path, map_location=device)

    model = Seq2SeqForecaster(
        input_dim=config.get("input_dim", input_dim),
        hidden_dim=config.get("hidden_dim", hidden_dim),
        horizon=config.get("horizon", horizon),
    ).to(device)

    if device.type == "cuda":
        model = model.to(torch.bfloat16)

    model.load_state_dict(state)
    model.eval()
    return model


def per_step_rmse(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    sq = (preds - targets) ** 2
    return np.sqrt(np.mean(sq, axis=2))

# Pick strides based on tolerance
def pick_strides(step_rmse: np.ndarray, tolerance: float) -> np.ndarray:
    strides = np.ones(step_rmse.shape[0], dtype=np.int32)
    for i, row in enumerate(step_rmse):
        under_tol = np.where(row <= tolerance)[0]
        if under_tol.size:
            strides[i] = under_tol.max() + 1
    return strides


def summarize_strides(strides: np.ndarray) -> str:
    counts = Counter(strides.tolist())
    parts = [f"{s}: {c}" for s, c in sorted(counts.items())]
    return ", ".join(parts)


def load_norm_stats(model_root: Path, input_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    norm_path = model_root / "norm_stats.npz"
    if norm_path.exists():
        print("Loaded norm_stats.npz")
        stats = np.load(norm_path)
        mean = stats["mean"].reshape(-1)
        std = stats["std"].reshape(-1)
        if mean.size >= input_dim and std.size >= input_dim:
            return mean[:input_dim].astype(np.float32), std[:input_dim].astype(np.float32)
    print(f"file not found: {model_root}/norm_stats.npz")
    return np.zeros(input_dim, dtype=np.float32), np.ones(input_dim, dtype=np.float32)


def advance_buffer(
    buffer: np.ndarray,
    preds: np.ndarray,
    stride: int,
    disp_mean: np.ndarray,
    disp_std: np.ndarray,
) -> None:
    if stride < 1:
        raise ValueError("invalid stride")
    seq_len = buffer.shape[1]
    input_dim = buffer.shape[2]
    step_count = min(stride, preds.shape[1])
    pred_steps = preds[:, :step_count, :]
    norm_disp = (pred_steps - disp_mean) / disp_std
    new_steps = np.zeros((buffer.shape[0], step_count, input_dim), dtype=buffer.dtype)
    new_steps[:, :, : norm_disp.shape[2]] = norm_disp

    if stride >= seq_len:
        buffer[:] = 0.0
        tail = new_steps[:, -seq_len:, :]
        buffer[:, :, :tail.shape[2]] = tail
    else:
        buffer[:, :-stride, :] = buffer[:, stride:, :]
        buffer[:, -stride:, :new_steps.shape[2]] = new_steps
def write_results_csv(rows: List[dict], path: Path) -> Path:
    num_models = None
    for r in rows:
        if r.get("section") == "run" and r.get("name") == "num_models_loaded":
            try:
                num_models = int(r.get("value"))
            except Exception:
                num_models = None
            break

    out_path = path

    if num_models is not None:
        if out_path.suffix.lower() == ".csv":
            tag = f"n{num_models}"
            if tag not in out_path.stem:
                out_path = out_path.with_name(f"{out_path.stem}_{tag}{out_path.suffix}")
        elif out_path.suffix == "":
            out_path = out_path.with_name(f"{out_path.name}_n{num_models}.csv")

    if out_path.exists():
        ts = time.time_ns()
        if out_path.suffix:
            out_path = out_path.with_name(f"{out_path.stem}_{ts}{out_path.suffix}")
        else:
            out_path = Path(str(out_path) + f"_{ts}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "name", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "section": row.get("section", ""),
                    "name": row.get("name", ""),
                    "value": row.get("value", ""),
                }
            )

    return out_path


def compute_channel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> List[dict]:
    channels = ["u", "v", "w"]
    metrics: List[dict] = []
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    for idx, ch in enumerate(channels):
        r2 = r2_score(y_true_flat[:, idx], y_pred_flat[:, idx])
        mse = mean_squared_error(y_true_flat[:, idx], y_pred_flat[:, idx])
        mae = mean_absolute_error(y_true_flat[:, idx], y_pred_flat[:, idx])
        metrics.append({"channel": ch, "r2": r2, "mse": mse, "mae": mae})
    return metrics


@dataclass
class InferenceTiming:
    total_seconds: float          # end-to-end: transfer + forward + cpu copy/concat + numpy
    forward_seconds: float        # forward-only
    samples: int
    batches: int

    @property
    def total_ms_per_sample(self) -> float:
        return (self.total_seconds * 1000.0) / max(self.samples, 1)

    @property
    def forward_ms_per_sample(self) -> float:
        return (self.forward_seconds * 1000.0) / max(self.samples, 1)

    @property
    def total_samples_per_sec(self) -> float:
        return self.samples / max(self.total_seconds, 1e-12)

    @property
    def forward_samples_per_sec(self) -> float:
        return self.samples / max(self.forward_seconds, 1e-12)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def batched_predict_timed(
    model: Seq2SeqForecaster,
    x_val: np.ndarray,
    device: torch.device,
    batch_size: int,
    warmup_batches: int = 1,
    repeats: int = 1,
    store_outputs: bool = True,
) -> Tuple[Optional[np.ndarray], InferenceTiming]:
    """
    Runs a FULL batched pass over x_val and returns timing for one pass
    and repeats for noise reduction.
    """
    assert repeats >= 1
    dtype = model.proj.weight.dtype
    n = len(x_val)
    num_batches = (n + batch_size - 1) // batch_size

    # Warm-up
    with torch.inference_mode():
        for b in range(min(warmup_batches, num_batches)):
            start = b * batch_size
            xb = torch.from_numpy(x_val[start : start + batch_size]).to(device=device, dtype=dtype)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(xb)
            else:
                _ = model(xb)
        _sync_if_cuda(device)

    def one_full_pass(store_outputs: bool) -> Tuple[Optional[np.ndarray], InferenceTiming]:
        outputs: Optional[List[torch.Tensor]] = [] if store_outputs else None
        forward_seconds = 0.0
        cuda_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []

        _sync_if_cuda(device)
        t0 = time.perf_counter()

        with torch.inference_mode():
            for start in range(0, n, batch_size):
                print()
                xb = torch.from_numpy(x_val[start : start + batch_size]).to(device=device, dtype=dtype)

                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(xb)
                    end_event.record()
                    cuda_events.append((start_event, end_event))
                    if store_outputs:
                        assert outputs is not None
                        outputs.append(out.float().cpu())
                else:
                    f0 = time.perf_counter()
                    out = model(xb)
                    f1 = time.perf_counter()
                    forward_seconds += (f1 - f0)
                    if store_outputs:
                        assert outputs is not None
                        outputs.append(out.cpu())

        _sync_if_cuda(device)
        t1 = time.perf_counter()

        total_seconds = (t1 - t0)
        if device.type == "cuda":
            _sync_if_cuda(device)
            forward_seconds = sum(
                start_event.elapsed_time(end_event) for start_event, end_event in cuda_events
            ) / 1000.0

        preds_np: Optional[np.ndarray] = None
        if store_outputs:
            assert outputs is not None
            preds_np = torch.cat(outputs, dim=0).numpy()

        return preds_np, InferenceTiming(
            total_seconds=total_seconds,
            forward_seconds=forward_seconds,
            samples=n,
            batches=num_batches,
        )

    preds, timing_first = one_full_pass(store_outputs=store_outputs)

    # Additional timing passe(s)
    if repeats > 1:
        totals = [timing_first.total_seconds]
        forwards = [timing_first.forward_seconds]
        for _ in range(repeats - 1):
            _, t = one_full_pass(store_outputs=False)
            totals.append(t.total_seconds)
            forwards.append(t.forward_seconds)

        tot_avg = float(np.mean(totals))
        tot_std = float(np.std(totals))
        fwd_avg = float(np.mean(forwards))
        fwd_std = float(np.std(forwards))
        print(
            f"  Timing repeats={repeats}: "
            f"total={tot_avg:.4f}s ± {tot_std:.4f}s | "
            f"forward={fwd_avg:.4f}s ± {fwd_std:.4f}s"
        )

    if store_outputs:
        assert preds is not None
    return preds, timing_first


@dataclass
class ShardResult:
    shard_idx: int
    device: torch.device
    preds: Optional[np.ndarray]
    targets: Optional[np.ndarray]
    timing: InferenceTiming

def load_model_replicas(
    model_root: Path,
    device_ids: List[int],
    input_dim: int,
    horizon: int,
    num_models: int,
) -> List[Tuple[torch.device, Seq2SeqForecaster]]:
    #Load a fixed number of replicas, round-robining across provided CUDA devices.
    if num_models < 1:
        raise ValueError("num_models must be >= 1.")

    devices: List[torch.device]
    if device_ids:
        devices = [torch.device(f"cuda:{d}") for d in device_ids]
    else:
        devices = [torch.device("cpu")]

    models: List[Tuple[torch.device, Seq2SeqForecaster]] = []
    for idx in range(num_models):
        device = devices[idx % len(devices)]
        model = load_checkpoint(model_root, device=device, input_dim=input_dim, horizon=horizon)
        models.append((device, model))

    return models

def run_shard(
    shard_idx: int,
    model: Seq2SeqForecaster,
    device: torch.device,
    X_shard: np.ndarray,
    y_shard: np.ndarray,
    batch_size: int,
    warmup_batches: int,
    timing_repeats: int,
    store_outputs: bool = True,
) -> ShardResult:
    preds, timing = batched_predict_timed(
        model=model,
        x_val=X_shard,
        device=device,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        repeats=timing_repeats,
        store_outputs=store_outputs,
    )
    return ShardResult(
        shard_idx=shard_idx,
        device=device,
        preds=preds,
        targets=y_shard if store_outputs else None,
        timing=timing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tolerance-driven surrogate evaluation")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT, help="Path to model root containing checkpoints")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset directory",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="inference batch size")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="per-step RMSE tolerance")
    parser.add_argument("--limit", type=int, default=None, help="limit on number of validation samples.")
    parser.add_argument("--timing-repeats", type=int, default=DEFAULT_TIMING_REPEATS, help="number of full-pass timing repeats")
    parser.add_argument("--timing-warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES, help="warm-up batches")
    parser.add_argument(
        "--num-models",
        type=int,
        default=DEFAULT_NUM_MODELS,
        help="number of model replicas to load",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV summary path",
    )
    parser.add_argument("--device-ids", type=int, nargs="+", default=None, help="CUDA device ids to use")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum concurrent shards.",
    )
    args = parser.parse_args()

    model_root = Path(args.model_root)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else model_root / "datasets"

    X_val = np.load(dataset_dir / "X_val.npy")
    y_val = np.load(dataset_dir / "y_val.npy")
    if args.limit:
        X_val = X_val[: args.limit]
        y_val = y_val[: args.limit]

    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    all_device_ids = list(range(torch.cuda.device_count())) if gpu_available else []
    device_ids = args.device_ids if args.device_ids is not None else all_device_ids
    device_ids = [int(d) for d in device_ids]

    horizon = y_val.shape[1]
    input_dim = X_val.shape[-1]

    if args.num_models is None:
        requested_models = len(device_ids) if gpu_available else 1
        auto_models = True
    else:
        if args.num_models < 1:
            raise ValueError("num_models must be >= 1.")
        requested_models = args.num_models
        auto_models = False

    target_models = min(requested_models, len(X_val))
    
    if not gpu_available:
        print("GPU not available")
        models = [
            (torch.device("cpu"), load_checkpoint(model_root, torch.device("cpu"), input_dim=input_dim, horizon=horizon))
            for _ in range(target_models)
        ]
    else:
        models = load_model_replicas(
            model_root=model_root,
            device_ids=device_ids,
            input_dim=input_dim,
            horizon=horizon,
            num_models=target_models,
        )


    num_models_loaded = len(models)
    num_devices_used = len({m[0] for m in models})
    print(
        f"Loaded {num_models_loaded} model replica(s) across {num_devices_used} device(s): "
        f"{', '.join(str(m[0]) for m in models)}"
    )

    max_surrogates = args.batch_size * num_models_loaded
    num_surrogates = min(len(X_val), max_surrogates)

    num_shards = min(num_models_loaded, (num_surrogates + args.batch_size - 1) // args.batch_size)
    models = models[:num_shards]
    active_devices = [m[0] for m in models]
    num_devices_used = len(set(active_devices))
    default_workers = num_devices_used or 1
    worker_count = min(num_shards, args.max_workers) if args.max_workers else min(num_shards, default_workers)
    print(f"Using {worker_count} worker(s)")

    print(
        f"Validation samples: {len(X_val)}, horizon: {horizon} "
        f"Tolerance={args.tolerance}, batch_size={args.batch_size}"
    )
    print(f"Surrogates per window={num_surrogates}")
    print(f"Timing: repeats={args.timing_repeats}, warmup_batches={args.timing_warmup_batches}")

    feat_mean, feat_std = load_norm_stats(model_root, input_dim)
    disp_mean = feat_mean[:3]
    disp_std = feat_std[:3]

    buffer = X_val[:num_surrogates].copy()
    targets_init = y_val[:num_surrogates]

    shard_slices: List[Tuple[int, int]] = []
    for shard_idx in range(num_shards):
        start = shard_idx * args.batch_size
        end = min(start + args.batch_size, num_surrogates)
        if start >= end:
            break
        shard_slices.append((start, end))

    shard_results_first: Optional[List[ShardResult]] = None
    timing_totals: Dict[int, dict] = {}

    def accumulate_timings(results: List[ShardResult]) -> None:
        for r in results:
            totals = timing_totals.setdefault(
                r.shard_idx,
                {
                    "device": r.device,
                    "total_seconds": 0.0,
                    "forward_seconds": 0.0,
                    "samples": 0,
                    "batches": 0,
                },
            )
            totals["device"] = r.device
            totals["total_seconds"] += r.timing.total_seconds
            totals["forward_seconds"] += r.timing.forward_seconds
            totals["samples"] += r.timing.samples
            totals["batches"] += r.timing.batches

    def run_window(
        warmup_batches: int,
        timing_repeats: int,
        pool: Optional[ThreadPoolExecutor],
    ) -> List[ShardResult]:
        window_results: List[ShardResult] = []
        if pool is None:
            for shard_idx, (device, model) in enumerate(models):
                start, end = shard_slices[shard_idx]
                window_results.append(
                    run_shard(
                        shard_idx=shard_idx,
                        model=model,
                        device=device,
                        X_shard=buffer[start:end],
                        y_shard=targets_init[start:end],
                        batch_size=args.batch_size,
                        warmup_batches=warmup_batches,
                        timing_repeats=timing_repeats,
                        store_outputs=True,
                    )
                )
        else:
            futures = []
            for shard_idx, (device, model) in enumerate(models):
                start, end = shard_slices[shard_idx]
                futures.append(
                    pool.submit(
                        run_shard,
                        shard_idx,
                        model,
                        device,
                        buffer[start:end],
                        targets_init[start:end],
                        args.batch_size,
                        warmup_batches,
                        timing_repeats,
                        True,
                    )
                )
            window_results = [fut.result() for fut in futures]
        window_results.sort(key=lambda r: r.shard_idx)
        return window_results

    def compute_stride_from_results(
        results: List[ShardResult],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        results.sort(key=lambda r: r.shard_idx)
        preds_local = np.concatenate([r.preds for r in results], axis=0)
        targets_local = np.concatenate([r.targets for r in results], axis=0)

        usable_h = min(preds_local.shape[1], horizon)
        preds_local = preds_local[:, :usable_h]
        targets_local = targets_local[:, :usable_h]

        step_rmse = per_step_rmse(preds_local, targets_local)
        strides_local = pick_strides(step_rmse, args.tolerance)
        recommended = int(np.median(strides_local))
        stride_local = max(1, min(recommended, usable_h))
        return preds_local, targets_local, strides_local, recommended, stride_local, usable_h

    def update_buffer_with_results(results: List[ShardResult], stride_steps: int) -> None:
        for r in results:
            if r.preds is None:
                continue
            start, end = shard_slices[r.shard_idx]
            advance_buffer(buffer[start:end], r.preds, stride_steps, disp_mean, disp_std)

    overall_start = time.perf_counter()
    if num_shards == 1 or worker_count == 1:
        window_results = run_window(args.timing_warmup_batches, args.timing_repeats, None)
        shard_results_first = window_results
        accumulate_timings(window_results)

        preds, targets, strides, recommended_stride, stride_steps, usable_horizon = compute_stride_from_results(window_results)
        window_count = (len(X_val) + stride_steps - 1) // stride_steps

        update_buffer_with_results(window_results, stride_steps)
        for _ in range(1, window_count):
            window_results = run_window(0, 1, None)
            accumulate_timings(window_results)
            update_buffer_with_results(window_results, stride_steps)
        overall_wall = time.perf_counter() - overall_start
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            window_results = run_window(args.timing_warmup_batches, args.timing_repeats, pool)
            shard_results_first = window_results
            accumulate_timings(window_results)

            preds, targets, strides, recommended_stride, stride_steps, usable_horizon = compute_stride_from_results(window_results)
            window_count = (len(X_val) + stride_steps - 1) // stride_steps

            update_buffer_with_results(window_results, stride_steps)
            for _ in range(1, window_count):
                window_results = run_window(0, 1, pool)
                accumulate_timings(window_results)
                update_buffer_with_results(window_results, stride_steps)
            overall_wall = time.perf_counter() - overall_start

    print(f"Stride steps (median)={stride_steps}, windows={window_count}")

    total_samples = num_surrogates
    total_samples_processed = total_samples * window_count

    timed_shards: List[ShardResult] = []
    for shard_idx, totals in timing_totals.items():
        timing = InferenceTiming(
            total_seconds=totals["total_seconds"],
            forward_seconds=totals["forward_seconds"],
            samples=totals["samples"],
            batches=totals["batches"],
        )
        timed_shards.append(
            ShardResult(
                shard_idx=shard_idx,
                device=totals["device"],
                preds=None,
                targets=None,
                timing=timing,
            )
        )
    timed_shards.sort(key=lambda r: r.shard_idx)

    agg_total = sum(r.timing.total_seconds for r in timed_shards)
    agg_forward = sum(r.timing.forward_seconds for r in timed_shards)

    print("\n=== Inference timing ===")
    for r in timed_shards:
        t = r.timing
        print(
            f"Shard {r.shard_idx} | device={r.device} | samples={t.samples} | "
            f"total={t.total_seconds:.4f}s ({t.total_ms_per_sample:.4f} ms/sample) | "
            f"forward={t.forward_seconds:.4f}s ({t.forward_ms_per_sample:.4f} ms/sample)"
        )
    print(
        f"Aggregate wall-clock (all windows, parallel per window): {overall_wall:.4f} s | "
        f"{(total_samples_processed / max(overall_wall, 1e-12)):.2f} samples/s"
    )
    print(
        f"Sum of per-shard totals (all windows): {agg_total:.4f} s | "
        f"sum of forward-only: {agg_forward:.4f} s"
    )

    channel_metrics = compute_channel_metrics(targets, preds)
    print("\n=== Channel metrics ===")
    for m in channel_metrics:
        print(f"  {m['channel']}: R^2={m['r2']:.4f} | MSE={m['mse']:.6f} | MAE={m['mae']:.6f}")

    step_rmse = per_step_rmse(preds, targets)
    strides = pick_strides(step_rmse, args.tolerance)
    recommended_stride = int(np.median(strides))
    print("\n=== Stride selection ===")
    print(f"  Selected stride/window per sample (counts): {summarize_strides(strides)}")
    print(f"  Recommended stride (median): {recommended_stride} | min={strides.min()} max={strides.max()}")
    if window_count > 1:
        print("  Note: stride statistics computed from the first window only.")
    print("  Interpretation: stride k means reusing the k-th output as the next starting point.")

    rows: List[dict] = [
        {"section": "run", "name": "model_root", "value": str(model_root)},
        {"section": "run", "name": "dataset_dir", "value": str(dataset_dir)},
        {"section": "run", "name": "num_models_loaded", "value": num_models_loaded},
        {"section": "run", "name": "num_devices_used", "value": num_devices_used},
        {"section": "run", "name": "max_workers", "value": worker_count},
        {"section": "data", "name": "samples", "value": total_samples},
        {"section": "data", "name": "window_steps", "value": window_count},
        {"section": "data", "name": "samples_processed", "value": total_samples_processed},
        {"section": "data", "name": "horizon", "value": usable_horizon},
        {"section": "data", "name": "tolerance", "value": args.tolerance},
        {"section": "data", "name": "batch_size", "value": args.batch_size},
        {
            "section": "timing",
            "name": "overall_wall_seconds",
            "value": f"{overall_wall:.6f}",
        },
        {
            "section": "timing",
            "name": "overall_samples_per_sec",
            "value": f"{(total_samples_processed / max(overall_wall, 1e-12)):.6f}",
        },
        {"section": "timing", "name": "sum_shard_total_seconds", "value": f"{agg_total:.6f}"},
        {"section": "timing", "name": "sum_shard_forward_seconds", "value": f"{agg_forward:.6f}"},
        {"section": "stride", "name": "counts", "value": summarize_strides(strides)},
        {"section": "stride", "name": "recommended_stride", "value": recommended_stride},
        {"section": "stride", "name": "stride_min", "value": int(strides.min())},
        {"section": "stride", "name": "stride_max", "value": int(strides.max())},
    ]

    for m in channel_metrics:
        rows.extend(
            [
                {"section": "channel", "name": f"{m['channel']}_r2", "value": f"{m['r2']:.6f}"},
                {"section": "channel", "name": f"{m['channel']}_mse", "value": f"{m['mse']:.6f}"},
                {"section": "channel", "name": f"{m['channel']}_mae", "value": f"{m['mae']:.6f}"},
            ]
        )

    for r in timed_shards:
        t = r.timing
        rows.extend(
            [
                {"section": "shard", "name": f"{r.shard_idx}_device", "value": str(r.device)},
                {"section": "shard", "name": f"{r.shard_idx}_samples", "value": t.samples},
                {"section": "shard", "name": f"{r.shard_idx}_total_seconds", "value": f"{t.total_seconds:.6f}"},
                {"section": "shard", "name": f"{r.shard_idx}_forward_seconds", "value": f"{t.forward_seconds:.6f}"},
            ]
        )

    write_results_csv(rows, args.output_csv)
    print(f"\nSaved CSV summary to {args.output_csv}")


if __name__ == "__main__":
    main()
