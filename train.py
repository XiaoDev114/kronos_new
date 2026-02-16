"""
Kronos 推理 GPT 训练脚本
========================
训练主模型，需要先运行 train_tokenizer.py 训练分词器。

使用方法:
    python train_tokenizer.py  # 先训练tokenizer
    python train.py            # 再训练主模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import json
from datetime import datetime

from config import ModelConfig, TokenizerConfig, TrainingConfig, PathConfig
from data_processor import get_dataloaders
from model.tokenizer import HierarchicalQuantizer
from model.kronos_reasoning import KronosReasoningGPT


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


def load_pretrained_tokenizer(device):
    """加载预训练的tokenizer。"""
    tokenizer_path = TrainingConfig.tokenizer_path
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"未找到预训练tokenizer: {tokenizer_path}\n"
            "请先运行: python train_tokenizer.py"
        )
    
    checkpoint = torch.load(tokenizer_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    tokenizer = HierarchicalQuantizer(**_build_tokenizer_kwargs(config)).to(device)
    
    tokenizer.load_state_dict(checkpoint['model_state_dict'], strict=False)
    tokenizer.eval()
    
    print(f"已加载预训练Tokenizer: {tokenizer_path}")
    print(f"Tokenizer训练轮数: {checkpoint['epoch']+1}, 损失: {checkpoint['loss']:.4f}")
    
    return tokenizer


def thought_regularization_loss(thought_states):
    """思维正则化损失，确保思维链的多样性和避免崩溃。"""
    if thought_states is None or thought_states.shape[0] < 2:
        return torch.tensor(0.0, device=thought_states.device if thought_states is not None else 'cpu')
    
    K, B, N, C = thought_states.shape
    diff = thought_states[1:] - thought_states[:-1]
    diversity_loss = torch.exp(-torch.mean(torch.norm(diff, dim=-1)))
    
    thought_flat = thought_states.view(K, B * N, C)
    collapse_loss = torch.exp(-torch.mean(torch.var(thought_flat, dim=1)))
    
    norm_var = torch.var(torch.norm(thought_states, dim=-1), dim=0)
    collapse_loss = collapse_loss + 0.1 * torch.exp(-torch.mean(norm_var))
    
    return TrainingConfig.diversity_weight * diversity_loss + TrainingConfig.collapse_weight * collapse_loss


def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, loss):
    """保存检查点。"""
    os.makedirs(PathConfig.checkpoint_dir, exist_ok=True)
    path = os.path.join(PathConfig.checkpoint_dir, f'checkpoint_epoch{epoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'tokenizer_state_dict': tokenizer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss
    }, path)
    print(f"检查点已保存: {path}")


def train_model():
    """训练模型主函数。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    tokenizer = load_pretrained_tokenizer(device)
    
    train_loader, val_loader, demo_loader, demo_dataset = get_dataloaders()
    
    model = KronosReasoningGPT(
        dim=ModelConfig.dim,
        depth=ModelConfig.depth,
        heads=ModelConfig.heads,
        num_thoughts=ModelConfig.num_thoughts,
        max_len=ModelConfig.max_len
    ).to(device)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=TrainingConfig.learning_rate, weight_decay=TrainingConfig.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainingConfig.epochs, eta_min=TrainingConfig.scheduler_eta_min)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'train_pred_loss': [], 'train_thought_loss': [], 'lr': []}
    best_val_loss = float('inf')
    
    print(f"\n开始训练 Kronos 推理 GPT...")
    print(f"训练轮数: {TrainingConfig.epochs}")
    print(f"批次大小: {TrainingConfig.batch_size}")
    print(f"学习率: {TrainingConfig.learning_rate}")
    print("-" * 50)
    
    for epoch in range(TrainingConfig.epochs):
        model.train()
        total_loss, total_pred, total_thought, num_batches = 0, 0, 0, 0
        
        for data, sector_ids, time_features in tqdm(train_loader, desc=f"第 {epoch+1}/{TrainingConfig.epochs} 轮"):
            data, sector_ids = data.to(device), sector_ids.to(device)
            
            with torch.no_grad():
                idx_coarse, idx_fine = tokenizer.encode(data)
            
            input_coarse, input_fine = idx_coarse[:, :-1], idx_fine[:, :-1]
            target_coarse, target_fine = idx_coarse[:, 1:], idx_fine[:, 1:]
            
            t_min = torch.clamp(time_features['minute'][:, :-1].to(device), 0, 239)
            t_day = torch.clamp(time_features['day'][:, :-1].to(device), 0, 30)
            t_month = torch.clamp(time_features['month'][:, :-1].to(device), 0, 11)
            t_year = torch.clamp(time_features['year'][:, :-1].to(device), 0, 99)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=False):
                logits_coarse, logits_fine, thoughts = model(input_coarse, input_fine, sector_ids, t_min, t_day, t_month, t_year)
                
                loss_c = criterion(logits_coarse.reshape(-1, logits_coarse.size(-1)), target_coarse.reshape(-1))
                loss_f = criterion(logits_fine.reshape(-1, logits_fine.size(-1)), target_fine.reshape(-1))
                pred_loss = loss_c + loss_f
                thought_loss = thought_regularization_loss(thoughts)
                loss = pred_loss + thought_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_pred += pred_loss.item()
            total_thought += thought_loss.item()
            num_batches += 1
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(total_loss / num_batches)
        history['train_pred_loss'].append(total_pred / num_batches)
        history['train_thought_loss'].append(total_thought / num_batches)
        history['lr'].append(current_lr)
        
        model.eval()
        val_loss, val_batches = 0, 0
        with torch.no_grad():
            for data, sector_ids, time_features in val_loader:
                data, sector_ids = data.to(device), sector_ids.to(device)
                idx_coarse, idx_fine = tokenizer.encode(data)
                
                t_min = torch.clamp(time_features['minute'][:, :-1].to(device), 0, 239)
                t_day = torch.clamp(time_features['day'][:, :-1].to(device), 0, 30)
                t_month = torch.clamp(time_features['month'][:, :-1].to(device), 0, 11)
                t_year = torch.clamp(time_features['year'][:, :-1].to(device), 0, 99)
                
                with torch.cuda.amp.autocast(enabled=False):
                    logits_coarse, logits_fine, _ = model(idx_coarse[:, :-1], idx_fine[:, :-1], sector_ids, t_min, t_day, t_month, t_year)
                    loss_c = criterion(logits_coarse.reshape(-1, logits_coarse.size(-1)), idx_coarse[:, 1:].reshape(-1))
                    loss_f = criterion(logits_fine.reshape(-1, logits_fine.size(-1)), idx_fine[:, 1:].reshape(-1))
                    val_loss += (loss_c + loss_f).item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)
        
        print(f"第 {epoch+1} 轮 - 训练损失: {history['train_loss'][-1]:.4f}, 验证损失: {avg_val_loss:.4f}, LR: {current_lr:.2e}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, avg_val_loss)
    
    history_path = os.path.join(PathConfig.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("-" * 50)
    print(f"训练完成! 最佳验证损失: {best_val_loss:.4f}")
    return model, tokenizer, demo_dataset


if __name__ == "__main__":
    train_model()
