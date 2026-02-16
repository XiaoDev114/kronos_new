import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import TokenizerConfig
from data_processor import get_dataloaders
from model.tokenizer import HierarchicalQuantizer


def _build_tokenizer_kwargs(config_dict=None):
    cfg = config_dict or {}
    return {
        "input_dim": cfg.get("input_dim", TokenizerConfig.input_dim),
        "hidden_dim": cfg.get("hidden_dim", TokenizerConfig.hidden_dim),
        "num_embeddings": cfg.get("num_embeddings", TokenizerConfig.num_embeddings),
        "embedding_dim": cfg.get("embedding_dim", TokenizerConfig.embedding_dim),
        "num_quantizers": cfg.get("num_quantizers", TokenizerConfig.num_quantizers),
        "commitment_cost": cfg.get("commitment_cost", TokenizerConfig.commitment_cost),
        "use_ema": cfg.get("use_ema", TokenizerConfig.use_ema),
        "ema_decay": cfg.get("ema_decay", TokenizerConfig.ema_decay),
        "ema_epsilon": cfg.get("ema_epsilon", TokenizerConfig.ema_epsilon),
        "restart_unused_codes": cfg.get(
            "restart_unused_codes", TokenizerConfig.restart_unused_codes
        ),
        "restart_every": cfg.get("restart_every", TokenizerConfig.restart_every),
        "restart_threshold": cfg.get("restart_threshold", TokenizerConfig.restart_threshold),
        "restart_noise_std": cfg.get("restart_noise_std", TokenizerConfig.restart_noise_std),
    }


def load_tokenizer(device):
    if not os.path.exists(TokenizerConfig.save_path):
        raise FileNotFoundError(f"Tokenizer not found: {TokenizerConfig.save_path}")

    checkpoint = torch.load(TokenizerConfig.save_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    tokenizer = HierarchicalQuantizer(**_build_tokenizer_kwargs(config)).to(device)
    tokenizer.load_state_dict(checkpoint["model_state_dict"], strict=False)
    tokenizer.eval()
    print(f"Tokenizer loaded from {TokenizerConfig.save_path}")
    return tokenizer, config


def _gini_from_counts(counts):
    total = counts.sum()
    if total <= 0:
        return 0.0

    sorted_counts = np.sort(counts.astype(np.float64))
    n = sorted_counts.size
    cumulative = np.cumsum(sorted_counts)
    gini = (n + 1 - 2.0 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def _distribution_metrics(indices, num_codes):
    counts = np.bincount(indices, minlength=num_codes).astype(np.int64)
    total = counts.sum()

    if total == 0:
        return {
            "unique_codes": 0,
            "usage_percent": 0.0,
            "dead_codes": num_codes,
            "dead_percent": 100.0,
            "perplexity": 0.0,
            "effective_usage_percent": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "top1_percent": 0.0,
            "top5_percent": 0.0,
            "gini": 0.0,
        }

    probs = counts / total
    active = counts > 0

    unique_codes = int(active.sum())
    usage_percent = unique_codes / num_codes * 100.0
    dead_codes = int(num_codes - unique_codes)
    dead_percent = dead_codes / num_codes * 100.0

    entropy = -np.sum(probs[active] * np.log(probs[active] + 1e-12))
    max_entropy = np.log(num_codes)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    perplexity = float(np.exp(entropy))
    effective_usage_percent = perplexity / num_codes * 100.0

    top1_percent = float(counts.max() / total * 100.0)
    k = min(5, num_codes)
    top5_percent = float(np.sort(counts)[-k:].sum() / total * 100.0)

    return {
        "unique_codes": unique_codes,
        "usage_percent": usage_percent,
        "dead_codes": dead_codes,
        "dead_percent": dead_percent,
        "perplexity": perplexity,
        "effective_usage_percent": effective_usage_percent,
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "top1_percent": top1_percent,
        "top5_percent": top5_percent,
        "gini": _gini_from_counts(counts),
    }


def evaluate_and_analyze(tokenizer, dataloader, num_codes, device):
    tokenizer.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_perp_levels = None
    num_batches = 0

    all_indices_levels = None

    use_amp = device.type == "cuda"

    print("\nStarting evaluation...")
    with torch.no_grad():
        for data, _, _ in tqdm(dataloader, desc="Evaluating"):
            data = data.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                _, x_recon, perplexities, level_indices = tokenizer(data, return_all=True)

            if total_perp_levels is None:
                total_perp_levels = [0.0 for _ in range(len(perplexities))]
                all_indices_levels = [[] for _ in range(len(level_indices))]

            total_mse += F.mse_loss(x_recon, data).item()
            total_mae += F.l1_loss(x_recon, data).item()
            for level_idx, perp in enumerate(perplexities):
                total_perp_levels[level_idx] += perp.item()
            for level_idx, idx in enumerate(level_indices):
                all_indices_levels[level_idx].append(idx.cpu().numpy().reshape(-1))
            num_batches += 1

    avg_mse = total_mse / max(num_batches, 1)
    avg_mae = total_mae / max(num_batches, 1)
    avg_perp_levels = [
        value / max(num_batches, 1) for value in (total_perp_levels or [0.0])
    ]

    level_metrics = {}
    for level_idx, level_indices in enumerate(all_indices_levels or []):
        merged_indices = (
            np.concatenate(level_indices) if level_indices else np.array([], dtype=np.int64)
        )
        level_metrics[f"level_{level_idx}"] = _distribution_metrics(merged_indices, num_codes)

    print("\nReconstruction Metrics:")
    print(f"MSE: {avg_mse:.6f}")
    print(f"MAE: {avg_mae:.6f}")

    print("\nBatch-level perplexity from forward pass:")
    for level_idx, perp in enumerate(avg_perp_levels):
        level_name = (
            "coarse"
            if level_idx == 0
            else "fine" if level_idx == 1 else f"residual_{level_idx}"
        )
        print(f"Level {level_idx} ({level_name}): {perp:.2f}")

    print("\nCodebook Usage (global distribution metrics):")
    severe_low_usage = False
    severe_low_eff_usage = False

    for level_name, metrics in level_metrics.items():
        print(f"[{level_name}]")
        print(
            f"Used={metrics['unique_codes']}/{num_codes} ({metrics['usage_percent']:.2f}%), "
            f"Dead={metrics['dead_codes']} ({metrics['dead_percent']:.2f}%)"
        )
        print(
            f"Perplexity={metrics['perplexity']:.2f} "
            f"(effective usage {metrics['effective_usage_percent']:.2f}%), "
            f"NormEntropy={metrics['normalized_entropy']:.4f}"
        )
        print(
            f"Top1={metrics['top1_percent']:.2f}%, Top5={metrics['top5_percent']:.2f}%, "
            f"Gini={metrics['gini']:.4f}"
        )

        severe_low_usage = severe_low_usage or metrics["usage_percent"] < 10
        severe_low_eff_usage = severe_low_eff_usage or metrics["effective_usage_percent"] < 10

    if severe_low_usage:
        print("WARNING: Severe low unique-usage detected.")

    if severe_low_eff_usage:
        print("WARNING: Severe low effective-usage (perplexity-based) detected.")

    metrics = {
        "reconstruction": {"mse": avg_mse, "mae": avg_mae},
        "perplexity_batch_avg_levels": avg_perp_levels,
        "codebook": {
            "num_codes": num_codes,
            **level_metrics,
        },
    }
    if "level_0" in level_metrics:
        metrics["codebook"]["coarse"] = level_metrics["level_0"]
    if "level_1" in level_metrics:
        metrics["codebook"]["fine"] = level_metrics["level_1"]

    metrics_path = os.path.join(os.path.dirname(TokenizerConfig.save_path), "tokenizer_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed metrics saved to {metrics_path}")

    return metrics


def visualize_reconstruction(tokenizer, dataloader, device, save_path="reconstruction_plot.png"):
    tokenizer.eval()
    data, _, _ = next(iter(dataloader))
    data = data.to(device)

    with torch.no_grad():
        _, x_recon, _, _, _, _ = tokenizer(data)

    original = data[0].cpu().numpy()
    reconstructed = x_recon[0].cpu().numpy()

    features = ["log_ret", "log_high", "log_low", "log_open", "log_vol", "log_amt"]

    plt.figure(figsize=(15, 10))
    for i in range(min(6, original.shape[1])):
        plt.subplot(3, 2, i + 1)
        plt.plot(original[:, i], label="Original", alpha=0.7)
        plt.plot(reconstructed[:, i], label="Reconstructed", alpha=0.7, linestyle="--")
        plt.title(features[i] if i < len(features) else f"Feature {i}")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nVisualization saved to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        tokenizer, ckpt_config = load_tokenizer(device)
    except FileNotFoundError:
        print(
            "Error: Pre-trained tokenizer not found. "
            "Please train it first using 'python train_tokenizer.py'."
        )
        return

    _, val_loader, _, _ = get_dataloaders()

    num_codes = ckpt_config.get("num_embeddings", TokenizerConfig.num_embeddings)
    evaluate_and_analyze(tokenizer, val_loader, num_codes=num_codes, device=device)

    visualize_reconstruction(tokenizer, val_loader, device)


if __name__ == "__main__":
    main()
