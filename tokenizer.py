import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TokenizerConfig


class VectorQuantizer(nn.Module):
    """Vector quantization with optional EMA codebook updates and dead-code restart."""

    def __init__(
        self,
        num_embeddings=TokenizerConfig.num_embeddings,
        embedding_dim=TokenizerConfig.embedding_dim,
        commitment_cost=TokenizerConfig.commitment_cost,
        use_ema=TokenizerConfig.use_ema,
        ema_decay=TokenizerConfig.ema_decay,
        ema_epsilon=TokenizerConfig.ema_epsilon,
        restart_unused_codes=TokenizerConfig.restart_unused_codes,
        restart_every=TokenizerConfig.restart_every,
        restart_threshold=TokenizerConfig.restart_threshold,
        restart_noise_std=TokenizerConfig.restart_noise_std,
    ):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost

        self._use_ema = use_ema
        self._ema_decay = ema_decay
        self._ema_epsilon = ema_epsilon
        self._restart_unused_codes = restart_unused_codes
        self._restart_every = max(1, int(restart_every))
        self._restart_threshold = float(restart_threshold)
        self._restart_noise_std = float(restart_noise_std)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1.0, 1.0)

        # Keep EMA buffers non-persistent so old checkpoints can still load.
        if self._use_ema:
            self.register_buffer("_ema_cluster_size", torch.zeros(self._num_embeddings), persistent=False)
            self.register_buffer("_ema_embed_sum", self._embedding.weight.data.clone(), persistent=False)
        else:
            self.register_buffer("_ema_cluster_size", torch.zeros(0), persistent=False)
            self.register_buffer("_ema_embed_sum", torch.zeros(0, 0), persistent=False)

        self._steps = 0

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self._embedding_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self._num_embeddings).type_as(flat_input)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        if self.training and self._use_ema:
            self._ema_update(flat_input, encodings)

        if self._use_ema:
            codebook_loss = flat_input.new_zeros(())
        else:
            codebook_loss = F.mse_loss(quantized, inputs.detach())

        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        loss = codebook_loss + self._commitment_cost * commitment_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        indices = encoding_indices.view(input_shape[0], input_shape[1])
        return loss, quantized, perplexity, indices

    def _ema_update(self, flat_input, encodings):
        with torch.no_grad():
            cluster_size = torch.sum(encodings, dim=0)
            embed_sum = torch.matmul(encodings.t(), flat_input)

            self._ema_cluster_size.mul_(self._ema_decay).add_(cluster_size, alpha=1 - self._ema_decay)
            self._ema_embed_sum.mul_(self._ema_decay).add_(embed_sum, alpha=1 - self._ema_decay)

            n = torch.sum(self._ema_cluster_size)
            smoothed_cluster_size = (
                (self._ema_cluster_size + self._ema_epsilon)
                / (n + self._num_embeddings * self._ema_epsilon)
                * n
            )

            normalized_embed = self._ema_embed_sum / smoothed_cluster_size.unsqueeze(1)
            self._embedding.weight.data.copy_(normalized_embed)

            self._steps += 1
            if self._restart_unused_codes and self._steps % self._restart_every == 0:
                self._restart_dead_codes(flat_input)

    def _restart_dead_codes(self, flat_input):
        if flat_input.numel() == 0:
            return

        dead_mask = self._ema_cluster_size < self._restart_threshold
        dead_indices = torch.nonzero(dead_mask, as_tuple=False).flatten()
        if dead_indices.numel() == 0:
            return

        random_ids = torch.randint(0, flat_input.shape[0], (dead_indices.numel(),), device=flat_input.device)
        new_codes = flat_input[random_ids]

        if self._restart_noise_std > 0:
            new_codes = new_codes + torch.randn_like(new_codes) * self._restart_noise_std

        self._embedding.weight.data[dead_indices] = new_codes
        self._ema_embed_sum[dead_indices] = new_codes
        self._ema_cluster_size[dead_indices] = self._restart_threshold

    def decode_ids(self, indices):
        """Decode code indices into embedding vectors."""
        return self._embedding(indices)

    def codebook_stats(self):
        """Expose coarse statistics for debugging."""
        if not self._use_ema:
            return {
                "use_ema": False,
                "steps": self._steps,
                "restart_unused_codes": self._restart_unused_codes,
            }

        active = self._ema_cluster_size > 0
        active_count = int(active.sum().item())
        return {
            "use_ema": True,
            "steps": self._steps,
            "active_codes": active_count,
            "dead_codes": int(self._num_embeddings - active_count),
            "restart_unused_codes": self._restart_unused_codes,
            "restart_threshold": self._restart_threshold,
        }


class HierarchicalQuantizer(nn.Module):
    """Residual vector quantizer with configurable depth."""

    def __init__(
        self,
        input_dim=TokenizerConfig.input_dim,
        hidden_dim=TokenizerConfig.hidden_dim,
        num_embeddings=TokenizerConfig.num_embeddings,
        embedding_dim=TokenizerConfig.embedding_dim,
        num_quantizers=TokenizerConfig.num_quantizers,
        commitment_cost=TokenizerConfig.commitment_cost,
        use_ema=TokenizerConfig.use_ema,
        ema_decay=TokenizerConfig.ema_decay,
        ema_epsilon=TokenizerConfig.ema_epsilon,
        restart_unused_codes=TokenizerConfig.restart_unused_codes,
        restart_every=TokenizerConfig.restart_every,
        restart_threshold=TokenizerConfig.restart_threshold,
        restart_noise_std=TokenizerConfig.restart_noise_std,
    ):
        super().__init__()
        self.num_quantizers = max(1, int(num_quantizers))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        quantizer_kwargs = {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "commitment_cost": commitment_cost,
            "use_ema": use_ema,
            "ema_decay": ema_decay,
            "ema_epsilon": ema_epsilon,
            "restart_unused_codes": restart_unused_codes,
            "restart_every": restart_every,
            "restart_threshold": restart_threshold,
            "restart_noise_std": restart_noise_std,
        }
        self.vq_layers = nn.ModuleList(
            [VectorQuantizer(**quantizer_kwargs) for _ in range(self.num_quantizers)]
        )
        # Backward-compatible module aliases for old checkpoints/scripts.
        self.vq_coarse = self.vq_layers[0]
        self.vq_fine = self.vq_layers[1] if self.num_quantizers > 1 else self.vq_layers[0]

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def _quantize_latent(self, z):
        residual = z
        z_q_total = torch.zeros_like(z)
        total_loss = z.new_zeros(())
        perplexities = []
        indices = []

        for quantizer in self.vq_layers:
            loss, z_q, perplexity, idx = quantizer(residual)
            total_loss = total_loss + loss
            z_q_total = z_q_total + z_q
            residual = residual - z_q
            perplexities.append(perplexity)
            indices.append(idx)

        return total_loss, z_q_total, perplexities, indices

    def forward(self, x, return_all=False):
        z = self.encoder(x)
        total_loss, z_q, perplexities, indices = self._quantize_latent(z)
        x_recon = self.decoder(z_q)

        if return_all:
            return total_loss, x_recon, perplexities, indices

        perp_c = perplexities[0]
        perp_f = perplexities[1] if self.num_quantizers > 1 else perplexities[0]
        idx_c = indices[0]
        idx_f = indices[1] if self.num_quantizers > 1 else indices[0]
        return total_loss, x_recon, perp_c, perp_f, idx_c, idx_f

    def encode_all(self, x):
        """Encode input into indices for every quantizer level."""
        z = self.encoder(x)
        _, _, _, indices = self._quantize_latent(z)
        return torch.stack(indices, dim=-1)

    def encode(self, x):
        """Backward-compatible two-stream encoding."""
        all_indices = self.encode_all(x)
        idx_c = all_indices[:, :, 0]
        idx_f = all_indices[:, :, 1] if self.num_quantizers > 1 else idx_c
        return idx_c, idx_f

    def decode_all(self, all_indices):
        """Decode all quantizer levels into reconstructed features."""
        if isinstance(all_indices, torch.Tensor):
            if all_indices.dim() != 3 or all_indices.size(-1) != self.num_quantizers:
                raise ValueError(
                    f"Expected indices tensor of shape [B, N, {self.num_quantizers}], "
                    f"got {tuple(all_indices.shape)}"
                )
            indices_per_level = [all_indices[:, :, i] for i in range(self.num_quantizers)]
        else:
            indices_per_level = list(all_indices)
            if len(indices_per_level) != self.num_quantizers:
                raise ValueError(
                    f"Expected {self.num_quantizers} levels, got {len(indices_per_level)}"
                )

        z_q = torch.zeros(
            indices_per_level[0].shape[0],
            indices_per_level[0].shape[1],
            self.vq_layers[0]._embedding_dim,
            device=indices_per_level[0].device,
        )

        for indices, quantizer in zip(indices_per_level, self.vq_layers):
            z_q = z_q + quantizer.decode_ids(indices)

        return self.decoder(z_q)

    def decode(self, idx_coarse, idx_fine):
        """Backward-compatible two-stream decoding."""
        if self.num_quantizers == 1:
            all_indices = torch.stack([idx_coarse], dim=-1)
            return self.decode_all(all_indices)

        levels = [idx_coarse, idx_fine]
        for _ in range(max(0, self.num_quantizers - 2)):
            levels.append(torch.zeros_like(idx_coarse))
        all_indices = torch.stack(levels, dim=-1)
        return self.decode_all(all_indices)

    def codebook_stats(self):
        stats = {f"level_{i}": q.codebook_stats() for i, q in enumerate(self.vq_layers)}
        if "level_0" in stats:
            stats["coarse"] = stats["level_0"]
        if "level_1" in stats:
            stats["fine"] = stats["level_1"]
        return stats
