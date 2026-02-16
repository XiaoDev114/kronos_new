"""Project configuration."""


class DataConfig:
    """Data processing settings."""

    data_dir = "dataset/"
    seq_len = 128
    demo_ratio = 0.05
    max_stocks = None
    train_val_split = 0.9
    outlier_sigma = 5
    stride_ratio = 0.5
    feature_cols = ["log_ret", "log_high", "log_low", "log_open", "log_vol", "log_amt"]
    base_year = 2010
    random_seed = 42


class TokenizerConfig:
    """Tokenizer (VQ-VAE) settings."""

    input_dim = 6
    hidden_dim = 128
    num_embeddings = 1024
    embedding_dim = 64
    num_quantizers = 2
    commitment_cost = 0.25

    # Step 1 stability improvements: EMA and dead-code restart.
    use_ema = True
    ema_decay = 0.99
    ema_epsilon = 1e-5
    restart_unused_codes = True
    restart_every = 200
    restart_threshold = 1.0
    restart_noise_std = 0.01

    epochs = 20
    learning_rate = 1e-3
    grad_clip = 1.0
    save_path = "checkpoints/tokenizer.pt"


class ModelConfig:
    """Kronos reasoning GPT settings."""

    dim = 256
    depth = 8
    heads = 8
    num_thoughts = 8
    max_len = 10000
    chunk_size = 8192
    dropout = 0.1
    vocab_size_coarse = 1024
    vocab_size_fine = 1024
    sector_vocab_size = 101


class TrainingConfig:
    """Training settings for Kronos reasoning GPT."""

    epochs = 20
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 0.01
    grad_clip = 1.0
    diversity_weight = 0.1
    collapse_weight = 0.1
    scheduler_T_max = epochs
    scheduler_eta_min = 1e-6
    save_dir = "checkpoints"
    tokenizer_path = "checkpoints/tokenizer.pt"


class EvaluationConfig:
    """Evaluation and prediction settings."""

    num_stocks = 100
    pred_steps = 30
    temperature = 1.0
    output_dir = "outputs"


class PathConfig:
    """Filesystem paths."""

    checkpoint_dir = "checkpoints"
    output_dir = "outputs"
    cache_file = "dataset_cache.pt"
