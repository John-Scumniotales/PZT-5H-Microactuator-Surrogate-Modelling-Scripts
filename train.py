import argparse
import contextlib
import csv
import math
import time
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Paths
DATA_DIR = Path("data")
DEFAULT_OUT_PATH = Path("Model")

# Hyperparameters
SEQ_LENGTH = 32          # past timesteps used by the encoder
PRED_HORIZON = 100         # how many future displacements to predict
BATCH_SIZE = 256
EPOCHS = 256
VAL_SPLIT = 0.1
LR = 3e-4
LR_FACTOR = 0.5
LR_PATIENCE = 5
MIN_DELTA = 1e-5
MIN_LR = 1e-6
GRAD_CLIP = 1.0
NUM_WORKERS = 2
INPUT_DROPOUT = 0.05       # dropout on encoder inputs
HIDDEN_DROPOUT = 0.1      # dropout on decoder hidden outputs
AUG_NOISE_STD = 0.005      # Gaussian noise on normalized inputs during training
R2_EVAL_EVERY = 5         # compute train/val R2 every N epochs
R2_MAX_TRAIN_BATCHES = 2500 # cap train batches for R2 to save time (None for full)
R2_MAX_VAL_BATCHES = None # use None to evaluate full val set
MAX_TRAIN_TIME = 23 * 3600  # seconds; stop training after this wall time


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_disp_raw(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Missing {path}")
    return pd.read_csv(
        path,
        sep=r"[,\s]+",
        comment="%",
        header=None,
        names=["time", name],
        engine="python",
        dtype={name: float},
    )


def load_int_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Missing {path}")
    return pd.read_csv(
        path,
        sep=r"[,\s]+",
        comment="%",
        header=None,
        names=["time", "int1", "int2"],
        engine="python",
        dtype={"int1": float, "int2": float},
    )


class Seq2SeqForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, out_dim: int = 3, num_layers: int = 2):
        super().__init__()
        dropout = 0.1 if num_layers > 1 else 0.0
        self.in_drop = nn.Dropout(INPUT_DROPOUT)
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.GRU(out_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dec_drop = nn.Dropout(HIDDEN_DROPOUT)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.horizon = horizon
        self.out_dim = out_dim

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        batch_size = src.size(0)
        src = self.in_drop(src)
        _, h = self.encoder(src)
        # start-of-sequence token is zeros in displacement space
        dec_in = torch.zeros(batch_size, 1, self.out_dim, device=src.device, dtype=src.dtype)
        outputs = []
        for t in range(self.horizon):
            dec_out, h = self.decoder(dec_in, h)
            step = self.proj(self.dec_drop(dec_out.squeeze(1)))
            outputs.append(step)
            dec_in = step.unsqueeze(1)
        return torch.stack(outputs, dim=1)


def prepare_data(
    output_norm: str,
    dataset_dir: Path,
    norm_path: Path,
    num_chunks: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    segments = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    feats_list: list[np.ndarray] = []
    disp_list: list[np.ndarray] = []

    for seg in segments:
        u_df = load_disp_raw(seg / "U_Table.txt", "u")
        v_df = load_disp_raw(seg / "V_Table.txt", "v")
        w_df = load_disp_raw(seg / "W_Table.txt", "w")
        it_df = load_int_raw(seg / "Int_Table.txt")

        n = len(u_df)
        if any(len(df) != n for df in (v_df, w_df, it_df)):
            sys.exit(f"Row count mismatch in segment {seg.name}")

        disp = np.stack([u_df.u.values, v_df.v.values, w_df.w.values], axis=1).astype(np.float32)
        ints = it_df[["int1", "int2"]].to_numpy(np.float32)
        feats = np.hstack([disp, ints])

        feats_list.append(feats)
        disp_list.append(disp)

    feats_all = np.vstack(feats_list)
    disp_all = np.vstack(disp_list)

    if not np.isfinite(feats_all).all() or not np.isfinite(disp_all).all():
        sys.exit("NaN/Inf found in data")

    N = feats_all.shape[0]
    print(f"Total timesteps: {N:,}")
    window_len = SEQ_LENGTH + PRED_HORIZON
    if num_chunks < 2:
        sys.exit("num_chunks must be at least 2")

    last_start = N - window_len
    if last_start < 0:
        sys.exit(f"Not enough steps for SEQ_LENGTH={SEQ_LENGTH} and HORIZON={PRED_HORIZON}")

    chunk_edges = np.linspace(0, N, num_chunks + 1, dtype=int)
    chunk_id = np.searchsorted(chunk_edges[1:], np.arange(N), side="right")
    starts = np.arange(0, last_start + 1, dtype=int)
    ends = starts + window_len - 1
    same_chunk = chunk_id[starts] == chunk_id[ends]
    valid_starts = starts[same_chunk]
    if valid_starts.size == 0:
        sys.exit("No valid windows within chunk boundaries; reduce num_chunks or window length.")

    rng = np.random.default_rng(42)
    chunk_perm = rng.permutation(num_chunks)
    val_chunks = set(chunk_perm[: max(1, int(num_chunks * VAL_SPLIT))])
    start_chunk_ids = chunk_id[valid_starts]
    train_mask = np.array([cid not in val_chunks for cid in start_chunk_ids], dtype=bool)
    train_starts = valid_starts[train_mask]
    val_starts = valid_starts[~train_mask]
    if train_starts.size == 0 or val_starts.size == 0:
        sys.exit("Empty train/val split after chunk assignment; adjust VAL_SPLIT or num_chunks.")

    X_train = np.zeros((train_starts.size, SEQ_LENGTH, feats_all.shape[1]), dtype=np.float32)
    y_train = np.zeros((train_starts.size, PRED_HORIZON, 3), dtype=np.float32)
    for i, start in enumerate(train_starts):
        X_train[i] = feats_all[start : start + SEQ_LENGTH]
        y_train[i] = disp_all[start + SEQ_LENGTH : start + window_len]

    X_val = np.zeros((val_starts.size, SEQ_LENGTH, feats_all.shape[1]), dtype=np.float32)
    y_val = np.zeros((val_starts.size, PRED_HORIZON, 3), dtype=np.float32)
    for i, start in enumerate(val_starts):
        X_val[i] = feats_all[start : start + SEQ_LENGTH]
        y_val[i] = disp_all[start + SEQ_LENGTH : start + window_len]

    feat_mean = X_train.mean(axis=(0, 1), keepdims=True)
    feat_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    if output_norm == "global":
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-6
    elif output_norm == "channel":
        y_mean = y_train.mean(axis=(0, 1), keepdims=True)
        y_std = y_train.std(axis=(0, 1), keepdims=True) + 1e-6
    else:
        raise ValueError(f"Unknown output_norm: {output_norm}")
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    np.save(dataset_dir / "X_train.npy", X_train)
    np.save(dataset_dir / "y_train.npy", y_train)
    np.save(dataset_dir / "X_val.npy", X_val)
    np.save(dataset_dir / "y_val.npy", y_val)
    np.savez(norm_path, mean=feat_mean, std=feat_std, y_mean=y_mean, y_std=y_std, output_norm=output_norm)

    print(f"Saved train/validation datasets to '{dataset_dir}'")
    return X_train, y_train, X_val, y_val, feat_mean, feat_std, y_mean, y_std


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    num_samples = 0
    use_amp = device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if AUG_NOISE_STD > 0:
                xb = xb + torch.randn_like(xb) * AUG_NOISE_STD
        with autocast_ctx():
            preds = model(xb)
            loss = criterion(preds, yb)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * xb.size(0)
        num_samples += xb.size(0)
    return total_loss / max(1, num_samples)


def evaluate_r2(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> float:
    model.eval()
    sse = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    n = 0
    batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            preds = model(xb)
            diff = preds - yb
            sse += diff.pow(2).sum().item()
            sum_y += yb.sum().item()
            sum_y2 += yb.pow(2).sum().item()
            n += yb.numel()
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    denom = sum_y2 - (sum_y * sum_y) / max(n, 1)
    if denom <= 0 or n == 0:
        return float("nan")
    return 1.0 - (sse / denom)


def save_log_row(
    log_csv_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    lr: float,
    train_r2: Optional[float] = None,
    val_r2: Optional[float] = None,
) -> None:
    header = ["epoch", "train_loss", "val_loss", "lr", "train_r2", "val_r2"]
    write_header = not log_csv_path.exists()
    with log_csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            lr,
            "" if train_r2 is None else train_r2,
            "" if val_r2 is None else val_r2,
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train seq2seq surrogate model.")
    parser.add_argument(
        "--output-norm",
        choices=["global", "channel"],
        default="global",
        help="Output standardization: global or channel-wise.",
    )
    parser.add_argument(
        "--out-path",
        default=str(DEFAULT_OUT_PATH),
        help="Output directory for checkpoints, datasets, and logs.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=24,
        help="Number of time chunks for leakage-safe split (windows stay within a chunk).",
    )
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_path = Path(args.out_path)
    ckpt_dir = out_path / "checkpoints"
    dataset_dir = out_path / "datasets"
    model_path = out_path / "seq2seq_surrogate.pt"
    best_ckpt_path = ckpt_dir / "best_seq2seq.pt"
    log_csv_path = out_path / "training_log.csv"
    norm_path = out_path / "norm_stats.npz"
    for p in (ckpt_dir, dataset_dir):
        p.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, feat_mean, feat_std, y_mean, y_std = prepare_data(
        args.output_norm,
        dataset_dir,
        norm_path,
        args.num_chunks,
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = Seq2SeqForecaster(input_dim=X_train.shape[-1], hidden_dim=64, horizon=PRED_HORIZON, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_FACTOR, patience=LR_PATIENCE, threshold=MIN_DELTA, min_lr=MIN_LR
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"Resumed from checkpoint: {best_ckpt_path}")

    best_val = math.inf
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, scaler)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, criterion, device, scaler)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        train_r2 = val_r2 = None
        if epoch % R2_EVAL_EVERY == 0:
            train_r2 = evaluate_r2(model, train_loader, device, max_batches=R2_MAX_TRAIN_BATCHES)
            val_r2 = evaluate_r2(model, val_loader, device, max_batches=R2_MAX_VAL_BATCHES)
        save_log_row(log_csv_path, epoch, train_loss, val_loss, lr, train_r2, val_r2)

        line = (
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
            f"| lr={lr:.2e}"
        )
        if train_r2 is not None and val_r2 is not None:
            line += f" | train_r2={train_r2:.4f} | val_r2={val_r2:.4f}"
        print(line)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"Saved new best checkpoint to {best_ckpt_path}")

        elapsed = time.time() - start_time
        if elapsed >= MAX_TRAIN_TIME:
            print(f"Time limit reached ({elapsed/3600:.2f} h), stopping early at epoch {epoch}.")
            break

    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    torch.save(
        {
            "model_state": model.state_dict(),
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "output_norm": args.output_norm,
            "config": {
                "seq_length": SEQ_LENGTH,
                "horizon": PRED_HORIZON,
                "input_dim": X_train.shape[-1],
                "hidden_dim": 64,
            },
        },
        model_path,
    )
    print(f"Saved final model to {model_path}")


if __name__ == "__main__":
    main()
