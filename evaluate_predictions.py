import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import warnings
import traceback
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from config import ModelConfig, TokenizerConfig, EvaluationConfig, PathConfig, DataConfig, TrainingConfig
from data_processor import AShareDataset
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


def load_model(device):
    """加载训练好的模型。"""
    tokenizer_path = TrainingConfig.tokenizer_path
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"未找到tokenizer: {tokenizer_path}")
    
    tokenizer_ckpt = torch.load(tokenizer_path, map_location=device, weights_only=False)
    config = tokenizer_ckpt['config']
    tokenizer = HierarchicalQuantizer(**_build_tokenizer_kwargs(config)).to(device)
    tokenizer.load_state_dict(tokenizer_ckpt['model_state_dict'], strict=False)
    tokenizer.eval()
    print(f"已加载Tokenizer: {tokenizer_path}")
    
    checkpoints = glob.glob(os.path.join(PathConfig.checkpoint_dir, 'checkpoint_*.pt'))
    if not checkpoints:
        raise FileNotFoundError("未找到检查点!")
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    checkpoint_path = checkpoints[0]
    print(f"加载检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = KronosReasoningGPT(
        dim=ModelConfig.dim,
        depth=ModelConfig.depth,
        heads=ModelConfig.heads,
        num_thoughts=ModelConfig.num_thoughts,
        max_len=ModelConfig.max_len
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer


def predict_sequence(model, tokenizer, features, sector_id, time_features, seq_stats, device):
    """自回归预测序列。"""
    model.eval()
    tokenizer.eval()
    
    with torch.no_grad():
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        sector_tensor = torch.tensor(sector_id, dtype=torch.long).unsqueeze(0).to(device)
        
        idx_coarse, idx_fine = tokenizer.encode(features)
        current_coarse, current_fine = idx_coarse.clone(), idx_fine.clone()
        
        t_min = torch.clamp(torch.tensor(time_features['minute'], dtype=torch.long).unsqueeze(0).to(device), 0, 239)
        t_day = torch.clamp(torch.tensor(time_features['day'], dtype=torch.long).unsqueeze(0).to(device), 0, 30)
        t_month = torch.clamp(torch.tensor(time_features['month'], dtype=torch.long).unsqueeze(0).to(device), 0, 11)
        t_year = torch.clamp(torch.tensor(time_features['year'], dtype=torch.long).unsqueeze(0).to(device), 0, 99)
        
        pred_coarse, pred_fine = [], []
        
        for _ in range(EvaluationConfig.pred_steps):
            curr_len = current_coarse.shape[1]
            logits_c, logits_f, _ = model(current_coarse, current_fine, sector_tensor, t_min[:, :curr_len], t_day[:, :curr_len], t_month[:, :curr_len], t_year[:, :curr_len])
            
            probs_c = torch.softmax(logits_c[:, -1, :] / EvaluationConfig.temperature, dim=-1)
            probs_f = torch.softmax(logits_f[:, -1, :] / EvaluationConfig.temperature, dim=-1)
            
            next_c = torch.multinomial(probs_c, num_samples=1)
            next_f = torch.multinomial(probs_f, num_samples=1)
            
            pred_coarse.append(next_c.item())
            pred_fine.append(next_f.item())
            
            current_coarse = torch.cat([current_coarse, next_c], dim=1)
            current_fine = torch.cat([current_fine, next_f], dim=1)
            t_min = torch.cat([t_min, t_min[:, -1:]], dim=1)
            t_day = torch.cat([t_day, t_day[:, -1:]], dim=1)
            t_month = torch.cat([t_month, t_month[:, -1:]], dim=1)
            t_year = torch.cat([t_year, t_year[:, -1:]], dim=1)
        
        all_coarse = idx_coarse[0].cpu().numpy().tolist() + pred_coarse
        all_fine = idx_fine[0].cpu().numpy().tolist() + pred_fine
        
        decoded = tokenizer.decode(
            torch.tensor([all_coarse], dtype=torch.long, device=device),
            torch.tensor([all_fine], dtype=torch.long, device=device),
        )
        decoded = decoded[0].cpu().numpy()

        # 反归一化回原始特征尺度（log_ret/log_high/...）
        mean = np.asarray(seq_stats['mean'], dtype=np.float32)
        std = np.asarray(seq_stats['std'], dtype=np.float32)
        decoded = decoded * std + mean
        return decoded


def evaluate_predictions(model, tokenizer, demo_dataset, device):
    """评估多支股票的预测效果。"""
    results = []
    symbols = list(demo_dataset.raw_data.keys())
    print(f"评估 {len(symbols)} 支股票 (全量 dataset)...")
    
    if len(symbols) == 0:
        print("警告: raw_data 为空!")
        return results
    
    skip_reasons = {'no_sample_idx': 0, 'short_data': 0, 'error': 0}
    
    for symbol in tqdm(symbols, desc="预测中"):
        raw_data = demo_dataset.raw_data[symbol]
        actual_closes = raw_data['close']
        sector_id = raw_data['sector_id']
        dates = raw_data['dates']
        
        sample_idx = next((i for i, s in enumerate(demo_dataset.symbols) if s == symbol), None)
        if sample_idx is None:
            skip_reasons['no_sample_idx'] += 1
            continue
            
        if len(actual_closes) < DataConfig.seq_len + EvaluationConfig.pred_steps:
            skip_reasons['short_data'] += 1
            continue
        
        try:
            pred_features = predict_sequence(
                model,
                tokenizer,
                demo_dataset.features[sample_idx],
                sector_id,
                demo_dataset.time_features[sample_idx],
                demo_dataset.seq_stats[sample_idx],
                device
            )
            
            hist_closes = actual_closes[:DataConfig.seq_len]
            hist_dates = dates[:DataConfig.seq_len]
            actual_future = np.array(actual_closes[DataConfig.seq_len:DataConfig.seq_len + EvaluationConfig.pred_steps])
            future_dates = dates[DataConfig.seq_len:DataConfig.seq_len + EvaluationConfig.pred_steps]
            pred_returns = pred_features[DataConfig.seq_len:DataConfig.seq_len + EvaluationConfig.pred_steps, 0]
            
            pred_closes = [hist_closes[-1]]
            for ret in pred_returns:
                pred_closes.append(pred_closes[-1] * np.exp(ret))
            pred_closes = np.array(pred_closes[1:])
            
            mape = np.mean(np.abs((actual_future - pred_closes) / (actual_future + 1e-6))) * 100
            rmse = np.sqrt(np.mean((actual_future - pred_closes) ** 2))
            dir_acc = np.mean((np.diff(actual_future) > 0) == (np.diff(pred_closes) > 0)) * 100
            
            results.append({
                'symbol': symbol, 
                'sector_id': sector_id, 
                'hist_closes': hist_closes,
                'hist_dates': hist_dates,
                'actual_future': actual_future, 
                'future_dates': future_dates,
                'pred_closes': pred_closes,
                'mape': mape, 
                'rmse': rmse, 
                'direction_accuracy': dir_acc
            })
        except Exception as e:
            skip_reasons['error'] += 1
            if skip_reasons['error'] <= 3:
                print(f"\n股票 {symbol} 预测出错: {e}")
                traceback.print_exc()
    
    print(f"\n跳过原因统计: {skip_reasons}")
    return results


def plot_metrics_summary(results):
    """生成全量股票的MAPE和方向准确率分布图。"""
    os.makedirs(PathConfig.output_dir, exist_ok=True)
    
    # 按MAPE排序，使图表更清晰（或者按代码排序）
    # 这里按MAPE排序，以便观察整体分布趋势
    results_sorted = sorted(results, key=lambda x: x['mape'])
    
    symbols = [r['symbol'] for r in results_sorted]
    mapes = [r['mape'] for r in results_sorted]
    dir_accs = [r['direction_accuracy'] for r in results_sorted]
    x = range(len(results))
    
    # 1. MAPE 分布图
    plt.figure(figsize=(24, 8))  # 增加宽度
    plt.plot(x, mapes, color='steelblue', linewidth=2, label='MAPE')
    plt.fill_between(x, 0, mapes, color='steelblue', alpha=0.3)
    plt.axhline(np.mean(mapes), color='red', linestyle='--', linewidth=1.5, label=f'平均 MAPE: {np.mean(mapes):.2f}%')
    plt.axhline(np.median(mapes), color='orange', linestyle='--', linewidth=1.5, label=f'中位数 MAPE: {np.median(mapes):.2f}%')
    
    plt.xlabel('个股 (按MAPE排序)', fontsize=12)
    plt.ylabel('MAPE (%)', fontsize=12)
    plt.title(f'全市场 {len(results)} 支股票预测 MAPE 分布', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.margins(x=0.01)
    
    save_path_mape = os.path.join(PathConfig.output_dir, 'all_stocks_mape.png')
    plt.savefig(save_path_mape, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"MAPE 分布图已保存: {save_path_mape}")

    # 2. 方向准确率分布图
    # 重新按方向准确率排序
    results_acc_sorted = sorted(results, key=lambda x: x['direction_accuracy'])
    dir_accs_sorted = [r['direction_accuracy'] for r in results_acc_sorted]
    
    plt.figure(figsize=(24, 8))  # 增加宽度
    plt.plot(x, dir_accs_sorted, color='seagreen', linewidth=2, label='方向准确率')
    plt.fill_between(x, 0, dir_accs_sorted, color='seagreen', alpha=0.3)
    plt.axhline(50, color='gray', linestyle=':', linewidth=1.5, label='随机基线 (50%)')
    plt.axhline(np.mean(dir_accs), color='red', linestyle='--', linewidth=1.5, label=f'平均准确率: {np.mean(dir_accs):.2f}%')
    
    plt.xlabel('个股 (按准确率排序)', fontsize=12)
    plt.ylabel('方向准确率 (%)', fontsize=12)
    plt.title(f'全市场 {len(results)} 支股票预测方向准确率分布', fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.margins(x=0.01)
    
    save_path_acc = os.path.join(PathConfig.output_dir, 'all_stocks_accuracy.png')
    plt.savefig(save_path_acc, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"方向准确率图已保存: {save_path_acc}")

    # 打印统计信息
    print(f"\n{'='*50}\n评估结果统计\n{'='*50}")
    print(f"股票总数: {len(results)}")
    print(f"MAPE - 均值: {np.mean(mapes):.2f}%, 中位数: {np.median(mapes):.2f}%")
    print(f"方向准确率 - 均值: {np.mean(dir_accs):.2f}%, >50%占比: {np.mean(np.array(dir_accs) > 50) * 100:.1f}%")
    print(f"{'='*50}")


def main():
    """主函数。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    model, tokenizer = load_model(device)
    demo_dataset = AShareDataset(mode='demo')
    results = evaluate_predictions(model, tokenizer, demo_dataset, device)
    
    if results:
        plot_metrics_summary(results)
        print("\n评估完成!")
    else:
        print("无有效预测!")


if __name__ == "__main__":
    main()
