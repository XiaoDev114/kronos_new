import json
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
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


def train_tokenizer(dataloader, tokenizer, device):
    optimizer = optim.Adam(tokenizer.parameters(), lr=TokenizerConfig.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TokenizerConfig.epochs, eta_min=1e-5
    )
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    tokenizer.train()
    history = {
        "loss": [],
        "vq_loss": [],
        "recon_loss": [],
        "perplexity_coarse": [],
        "perplexity_fine": [],
        "perplexity_levels": [],
        "dead_codes_coarse": [],
        "dead_codes_fine": [],
    }

    print("Training VQ-VAE Tokenizer...")
    print(f"Codebook size: {TokenizerConfig.num_embeddings}")
    print(f"Embedding dim: {TokenizerConfig.embedding_dim}")
    print(f"Num quantizers: {TokenizerConfig.num_quantizers}")
    print(f"Epochs: {TokenizerConfig.epochs}")
    print("-" * 50)

    best_loss = float("inf")

    for epoch in range(TokenizerConfig.epochs):
        tokenizer.train()
        total_loss = 0.0
        total_vq = 0.0
        total_recon = 0.0
        total_perp_levels = None
        num_batches = 0

        for batch_idx, (data, _, _) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{TokenizerConfig.epochs}")
        ):
            data = data.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                vq_loss, x_recon, perplexities, _ = tokenizer(data, return_all=True)
                recon_loss = F.mse_loss(x_recon, data)
                loss = recon_loss + vq_loss

            if total_perp_levels is None:
                total_perp_levels = [0.0 for _ in range(len(perplexities))]

            if torch.isnan(loss):
                print(f"Warning: NaN loss at batch {batch_idx}, skip.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), TokenizerConfig.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_vq += vq_loss.item()
            total_recon += recon_loss.item()
            for level_idx, perp in enumerate(perplexities):
                total_perp_levels[level_idx] += perp.item()
            num_batches += 1

        scheduler.step()

        denom = max(num_batches, 1)
        avg_loss = total_loss / denom
        avg_vq = total_vq / denom
        avg_recon = total_recon / denom
        avg_perp_levels = [v / denom for v in (total_perp_levels or [0.0])]
        avg_perp_c = avg_perp_levels[0]
        avg_perp_f = avg_perp_levels[1] if len(avg_perp_levels) > 1 else avg_perp_levels[0]

        history["loss"].append(avg_loss)
        history["vq_loss"].append(avg_vq)
        history["recon_loss"].append(avg_recon)
        history["perplexity_coarse"].append(avg_perp_c)
        history["perplexity_fine"].append(avg_perp_f)
        history["perplexity_levels"].append(avg_perp_levels)

        codebook_stats = tokenizer.codebook_stats()
        dead_c = codebook_stats["coarse"].get("dead_codes", -1)
        dead_f = codebook_stats["fine"].get("dead_codes", -1)
        history["dead_codes_coarse"].append(dead_c)
        history["dead_codes_fine"].append(dead_f)

        print(
            f"Epoch {epoch + 1} - loss: {avg_loss:.4f}, vq: {avg_vq:.4f}, "
            f"recon_mse: {avg_recon:.6f}, perp_c: {avg_perp_c:.1f}, perp_f: {avg_perp_f:.1f}"
        )
        if len(avg_perp_levels) > 2:
            level_text = ", ".join(
                [f"L{i}={value:.1f}" for i, value in enumerate(avg_perp_levels)]
            )
            print(f"  Perplexity by level: {level_text}")
        if dead_c >= 0 and dead_f >= 0:
            print(
                f"  EMA stats - dead codes coarse/fine: "
                f"{dead_c}/{TokenizerConfig.num_embeddings}, "
                f"{dead_f}/{TokenizerConfig.num_embeddings}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_tokenizer(tokenizer, history, epoch, avg_loss)

    print("-" * 50)
    print(f"Tokenizer training done. best_loss={best_loss:.4f}")
    return tokenizer, history


def save_tokenizer(tokenizer, history, epoch, loss):
    os.makedirs(os.path.dirname(TokenizerConfig.save_path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": tokenizer.state_dict(),
            "config": {
                "input_dim": TokenizerConfig.input_dim,
                "hidden_dim": TokenizerConfig.hidden_dim,
                "num_embeddings": TokenizerConfig.num_embeddings,
                "embedding_dim": TokenizerConfig.embedding_dim,
                "num_quantizers": TokenizerConfig.num_quantizers,
                "commitment_cost": TokenizerConfig.commitment_cost,
                "use_ema": TokenizerConfig.use_ema,
                "ema_decay": TokenizerConfig.ema_decay,
                "ema_epsilon": TokenizerConfig.ema_epsilon,
                "restart_unused_codes": TokenizerConfig.restart_unused_codes,
                "restart_every": TokenizerConfig.restart_every,
                "restart_threshold": TokenizerConfig.restart_threshold,
                "restart_noise_std": TokenizerConfig.restart_noise_std,
            },
            "history": history,
            "loss": loss,
        },
        TokenizerConfig.save_path,
    )

    print(f"Tokenizer saved: {TokenizerConfig.save_path}")


def load_tokenizer(device):
    if not os.path.exists(TokenizerConfig.save_path):
        raise FileNotFoundError(f"Tokenizer not found: {TokenizerConfig.save_path}")

    checkpoint = torch.load(TokenizerConfig.save_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    tokenizer = HierarchicalQuantizer(**_build_tokenizer_kwargs(config)).to(device)

    tokenizer.load_state_dict(checkpoint["model_state_dict"], strict=False)
    tokenizer.eval()

    print(f"Tokenizer loaded: {TokenizerConfig.save_path}")
    print(f"Trained epochs: {checkpoint['epoch'] + 1}, loss: {checkpoint['loss']:.4f}")

    return tokenizer


def evaluate_tokenizer(tokenizer, dataloader, device):
    tokenizer.eval()
    total_mse, total_mae, num_batches = 0.0, 0.0, 0
    use_amp = device.type == "cuda"

    print("\nEvaluating tokenizer reconstruction quality...")

    with torch.no_grad():
        for data, _, _ in tqdm(dataloader, desc="Evaluate"):
            data = data.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, x_recon, _, _, _, _ = tokenizer(data)

            total_mse += F.mse_loss(x_recon, data).item()
            total_mae += F.l1_loss(x_recon, data).item()
            num_batches += 1

    avg_mse = total_mse / max(num_batches, 1)
    avg_mae = total_mae / max(num_batches, 1)

    print(f"Reconstruction MSE: {avg_mse:.6f}")
    print(f"Reconstruction MAE: {avg_mae:.6f}")

    return avg_mse, avg_mae


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _, _ = get_dataloaders()

    tokenizer = HierarchicalQuantizer(**_build_tokenizer_kwargs()).to(device)

    print(f"\nTokenizer params: {sum(p.numel() for p in tokenizer.parameters()):,}")

    tokenizer, history = train_tokenizer(train_loader, tokenizer, device)
    evaluate_tokenizer(tokenizer, val_loader, device)

    history_path = os.path.join(
        os.path.dirname(TokenizerConfig.save_path), "tokenizer_history.json"
    )
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")


if __name__ == "__main__":
    main()
