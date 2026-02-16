import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DataConfig, PathConfig, TrainingConfig

warnings.filterwarnings("ignore")


class AShareDataset(Dataset):
    def __init__(self, mode="train"):
        self.seq_len = DataConfig.seq_len
        self.data_dir = DataConfig.data_dir
        self.mode = mode
        self.demo_ratio = DataConfig.demo_ratio
        self.max_stocks = DataConfig.max_stocks

        self.features = []
        self.sector_ids = []
        self.time_features = []
        self.seq_stats = []
        self.dates = []
        self.symbols = []
        self.raw_data = {}

        cache_file = PathConfig.cache_file.replace(".pt", f"_{mode}.pt")
        if self.max_stocks:
            cache_file = PathConfig.cache_file.replace(".pt", f"_{mode}_{self.max_stocks}.pt")

        cache_loaded = False
        if os.path.exists(cache_file):
            print(f"Loading {mode} dataset cache: {cache_file}")
            cached_data = torch.load(cache_file, weights_only=False)
            cached_seq_stats = cached_data.get("seq_stats")

            if cached_seq_stats is not None and len(cached_seq_stats) == len(
                cached_data.get("features", [])
            ):
                self.features = cached_data["features"]
                self.sector_ids = cached_data["sector_ids"]
                self.time_features = cached_data["time_features"]
                self.seq_stats = cached_seq_stats
                self.dates = cached_data.get("dates", [])
                self.symbols = cached_data.get("symbols", [])
                self.raw_data = cached_data.get("raw_data", {})
                cache_loaded = True
            else:
                print(f"{mode} cache is outdated (missing seq_stats). Rebuilding cache.")

        if not cache_loaded:
            print(f"Processing {mode} dataset from: {self.data_dir}")
            self._process_data()

            save_data = {
                "features": self.features,
                "sector_ids": self.sector_ids,
                "time_features": self.time_features,
                "seq_stats": self.seq_stats,
                "dates": self.dates,
                "symbols": self.symbols,
            }
            if mode == "demo":
                save_data["raw_data"] = self.raw_data

            print(f"Saving {mode} dataset cache: {cache_file}")
            torch.save(save_data, cache_file)

    def _process_data(self):
        files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        print(f"Found {len(files)} CSV files")

        if self.max_stocks:
            np.random.seed(DataConfig.random_seed)
            files = list(np.random.choice(files, min(self.max_stocks, len(files)), replace=False))
            print(f"Sampled {len(files)} files")

        all_stock_data = []
        for file_path in tqdm(files, desc="Loading stock data"):
            try:
                df = pd.read_csv(file_path)
                if "date" not in df.columns:
                    continue

                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                if "volume" in df.columns:
                    df = df[df["volume"] > 0].reset_index(drop=True)

                if len(df) < self.seq_len * 2:
                    continue

                if "symbol" in df.columns:
                    symbol = str(df["symbol"].iloc[0])
                else:
                    symbol = os.path.basename(file_path).split(".")[0]

                if "sector_id" in df.columns:
                    sector_id = int(df["sector_id"].iloc[0])
                else:
                    sector_id = self._get_sector_id(symbol)

                all_stock_data.append({"symbol": symbol, "sector_id": sector_id, "df": df})
            except Exception:
                continue

        if not all_stock_data:
            print("No valid stock data found.")
            return

        all_dates = sorted(set(d for stock in all_stock_data for d in stock["df"]["date"].tolist()))
        split_demo_idx = int(len(all_dates) * (1 - self.demo_ratio))
        split_train_val_idx = int(split_demo_idx * DataConfig.train_val_split)

        print(f"Date range: {all_dates[0]} to {all_dates[-1]}")

        if self.mode == "train":
            target_dates = set(all_dates[:split_train_val_idx])
            print(f"Train range: {all_dates[0]} to {all_dates[split_train_val_idx - 1]}")
        elif self.mode == "val":
            target_dates = set(all_dates[split_train_val_idx:split_demo_idx])
            print(f"Val range: {all_dates[split_train_val_idx]} to {all_dates[split_demo_idx - 1]}")
        else:
            target_dates = set(all_dates[split_demo_idx:])
            print(f"Demo range: {all_dates[split_demo_idx]} to {all_dates[-1]}")

        stride = max(1, int(self.seq_len * DataConfig.stride_ratio))

        for stock in tqdm(all_stock_data, desc=f"Processing {self.mode} data"):
            try:
                df = stock["df"]
                symbol = stock["symbol"]
                sector_id = stock["sector_id"]

                df_filtered = df[df["date"].isin(target_dates)].copy()
                if len(df_filtered) < self.seq_len:
                    continue

                df_filtered = df_filtered.sort_values("date").reset_index(drop=True)
                prev_close = df_filtered["close"].shift(1)

                df_filtered["log_ret"] = np.log(df_filtered["close"] / prev_close)
                df_filtered["log_high"] = np.log(df_filtered["high"] / prev_close)
                df_filtered["log_low"] = np.log(df_filtered["low"] / prev_close)
                df_filtered["log_open"] = np.log(df_filtered["open"] / prev_close)
                df_filtered["log_vol"] = np.log1p(df_filtered["volume"])
                df_filtered["log_amt"] = np.log1p(df_filtered["amount"])
                df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan)
                df_filtered = df_filtered.dropna(subset=DataConfig.feature_cols).reset_index(drop=True)

                if len(df_filtered) < self.seq_len:
                    continue

                data = df_filtered[DataConfig.feature_cols].values.astype(np.float32)
                if np.isnan(data).any() or np.isinf(data).any():
                    continue

                num_seqs = (len(data) - self.seq_len) // stride + 1
                if num_seqs <= 0:
                    continue

                if self.mode == "demo":
                    self.raw_data[symbol] = {
                        "dates": df_filtered["date"].tolist(),
                        "close": df_filtered["close"].tolist(),
                        "open": df_filtered["open"].tolist(),
                        "high": df_filtered["high"].tolist(),
                        "low": df_filtered["low"].tolist(),
                        "volume": df_filtered["volume"].tolist(),
                        "sector_id": sector_id,
                    }

                for i in range(num_seqs):
                    start_idx = i * stride
                    seq = data[start_idx : start_idx + self.seq_len]

                    mean = np.mean(seq, axis=0).astype(np.float32)
                    std = np.std(seq, axis=0).astype(np.float32)
                    std[std == 0] = 1.0
                    seq_norm = ((seq - mean) / std).astype(np.float32)

                    self.features.append(seq_norm)
                    self.sector_ids.append(sector_id)
                    self.seq_stats.append({"mean": mean, "std": std})

                    seq_dates = df_filtered["date"].iloc[start_idx : start_idx + self.seq_len].tolist()
                    self.time_features.append(self._extract_time_features(seq_dates))
                    if self.mode == "demo":
                        self.dates.append(seq_dates)
                        self.symbols.append(symbol)
            except Exception:
                continue

        print(f"Done: {self.mode} total sequences = {len(self.features)}")

    def _extract_time_features(self, dates):
        return {
            "minute": np.zeros(len(dates), dtype=np.int64),
            "day": np.clip(np.array([d.day for d in dates], dtype=np.int64) - 1, 0, 30),
            "month": np.clip(np.array([d.month for d in dates], dtype=np.int64) - 1, 0, 11),
            "year": np.clip(
                np.array([d.year - DataConfig.base_year for d in dates], dtype=np.int64), 0, 99
            ),
            "weekday": np.array([d.weekday() for d in dates], dtype=np.int64),
            "quarter": np.array([(d.month - 1) // 3 for d in dates], dtype=np.int64),
        }

    def _get_sector_id(self, symbol):
        sectors = {
            "banking": 0,
            "securities": 1,
            "insurance": 2,
            "real_estate": 3,
            "construction": 4,
            "steel": 5,
            "coal": 6,
            "petroleum": 7,
            "chemical": 8,
            "building_materials": 9,
            "nonferrous_metals": 10,
            "machinery": 11,
            "electrical_equipment": 12,
            "defense": 13,
            "automobile": 14,
            "home_appliance": 15,
            "light_manufacturing": 16,
            "agriculture": 17,
            "food_beverage": 18,
            "textile_apparel": 19,
            "medicine": 20,
            "biotech": 21,
            "medical_devices": 22,
            "electronics": 23,
            "semiconductor": 24,
            "computer": 25,
            "communication": 26,
            "media": 27,
            "internet": 28,
            "software": 29,
            "power": 30,
            "utilities": 31,
            "transportation": 32,
            "logistics": 33,
            "retail": 34,
            "commerce": 35,
            "tourism": 36,
            "hotel_restaurant": 37,
            "education": 38,
            "environmental": 39,
            "public_services": 40,
            "new_energy": 41,
            "new_materials": 42,
            "other": 50,
        }

        try:
            code = int(symbol) if symbol.isdigit() else 0
        except Exception:
            code = 0

        if 600000 <= code <= 600999:
            return sectors["banking"]
        if 601288 <= code <= 601398:
            return sectors["banking"]
        if 600030 <= code <= 600999:
            return sectors["securities"]
        if 300000 <= code <= 300749:
            return sectors["electronics"]
        if 300750 <= code <= 300999:
            return sectors["new_energy"]
        if 688000 <= code <= 688999:
            return sectors["semiconductor"]
        if 2000 <= code <= 2999:
            return sectors["machinery"]
        return sectors["other"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        sector_id = torch.tensor(self.sector_ids[idx], dtype=torch.long)
        time_feat = self.time_features[idx]
        time_tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in time_feat.items()}
        return features, sector_id, time_tensors


def collate_fn(batch):
    features, sector_ids, time_feats = zip(*batch)
    features = torch.stack(features, dim=0)
    sector_ids = torch.stack(sector_ids, dim=0)
    time_features = {k: torch.stack([t[k] for t in time_feats], dim=0) for k in time_feats[0].keys()}
    return features, sector_ids, time_features


def get_dataloaders():
    train_dataset = AShareDataset(mode="train")
    val_dataset = AShareDataset(mode="val")
    demo_dataset = AShareDataset(mode="demo")

    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainingConfig.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainingConfig.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    demo_loader = DataLoader(
        demo_dataset,
        batch_size=TrainingConfig.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, demo: {len(demo_dataset)}")
    return train_loader, val_loader, demo_loader, demo_dataset


if __name__ == "__main__":
    train_loader, _, _, _ = get_dataloaders()
    for batch_x, _, _ in train_loader:
        print("batch shape:", batch_x.shape)
        break
