# src/dataset.py
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split


def preprocess_paired_csv_to_npy(
    cache_dir,
    clinical_csv: str,
    gene_csv: str,
    id_col: str = "ID",
    target_cols: List[str] = None,                      # y: 临床中你要预测/条件化的列
    *,
    missing_value: float = -1.0,                        # 任意缺失值直接用-1代替
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    
    """
    从两个 CSV（临床 + 基因）通过 ID 配对 
    返回:
        x: np.ndarray, shape=(N, D_x) —— 基因数据（+ 可选临床特征）
        y: np.ndarray, shape=(N, D_y) —— 目标临床变量（用于监督/条件）
        meta: dict ，元信息（列名、ID 列表、shape 等）
    """
    print(f"🔗 Loading clinical data from {clinical_csv}")
    clinical_df = pd.read_csv(clinical_csv)
    print(f"🧬 Loading gene data from {gene_csv}")
    gene_df = pd.read_csv(gene_csv)

    if id_col not in clinical_df.columns:
        raise ValueError(f"'{id_col}' not found in clinical data. Columns: {list(clinical_df.columns)}")
    if id_col not in gene_df.columns:
        raise ValueError(f"'{id_col}' not found in gene data. Columns: {list(gene_df.columns)}")

    clinical_df[id_col] = clinical_df[id_col].astype(str)
    gene_df[id_col] = gene_df[id_col].astype(str)
    print("🪢 Inner joining on ID...")
    merged_df = clinical_df.merge(gene_df, on=id_col, how="inner")
    print(f"✅ After join: {len(merged_df)} samples (clinical: {len(clinical_df)}, gene: {len(gene_df)})")

    if target_cols is None:
        raise ValueError("target_cols must be specified)")
    missing_targets = set(target_cols) - set(clinical_df.columns)
    if missing_targets:
        raise ValueError(f"Target columns not in clinical data: {missing_targets}")
    y_df = merged_df[target_cols].copy()

    gene_cols = [col for col in gene_df.columns if col != id_col]
    x_gene_df = merged_df[gene_cols].copy()
    x_df = x_gene_df
    print(f"🧬 x shape: genes only ({x_df.shape[1]})")

    y_df.fillna(missing_value, inplace=True)

    x = x_df.values.astype(np.float32)
    y = y_df.values.astype(np.float32)

    meta = {
        "clinical_csv": str(clinical_csv),
        "gene_csv": str(gene_csv),
        "id_col": id_col,
        "n_samples": len(merged_df),
        "id_list": merged_df[id_col].tolist(),   
        "clinical_factor": target_cols,
        "missing_value_placeholder": missing_value,
        "join_info": {
            "clinical_before": len(clinical_df),
            "gene_before": len(gene_df),
            "after_join": len(merged_df)
        }
    }
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "x.npy", x)
    np.save(cache_dir / "y.npy", y)
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"💾 Cached to {cache_dir}")


def load_dataset_from_npy(
    seed,
    x_path: str = "x.npy",
    y_path: str = "y.npy",
    val_ratio: float = 0.05,
    missing_value: float = -1.0,
) -> Tuple[object, Optional[object]]:
    """
    Returns:
        train_dataset: obj with .x, .y
        val_dataset: obj with .x, .y, or None if no valid samples
    """
    x = np.load(x_path)
    y = np.load(y_path)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    is_valid = ~(y == missing_value).any(axis=1)   
    n_valid = is_valid.sum()
    print(f"📊Total samples: {len(x)} | Valid (y ≠ {missing_value}): {n_valid}")
    if n_valid == 0:
        raise ValueError(f"No valid samples found (all y == {missing_value})")
    val_size = max(1, int(n_valid * val_ratio)) 
    if n_valid < val_size:
        print(f"⚠️  Not enough valid samples for {val_ratio*100}% val. Using {n_valid} as val.")
        val_size = n_valid
    valid_idx = np.where(is_valid)[0]
    train_idx, val_idx = train_test_split(
        valid_idx,
        test_size=val_size,
        random_state=seed,
        shuffle=True
    )
    missing_idx = np.where(~is_valid)[0]
    train_idx = np.concatenate([train_idx, missing_idx])
    def make_dataset(indices):
        return SimpleDataset(x[indices], y[indices], missing_value=missing_value)  # ✅
    train_ds = make_dataset(train_idx)
    val_ds = make_dataset(val_idx) if len(val_idx) > 0 else None
    print(f"✅ Train: {len(train_ds.x)} samples | Val: {len(val_ds.x) if val_ds else 0}")
    return train_ds, val_ds

def apply_hvg_filter(
    x_path: str,
    cache_dir: str,
    n_hvg: int = 5000,
) -> str:
    """
    Apply HVG filtering by variance. Save x_hvg.npy and update meta.json.
    """
    cache_dir = Path(cache_dir)
    x = np.load(x_path)
    print(f"🧬 Loaded x: {x.shape} from {x_path}")
    if n_hvg <= 0 or n_hvg >= x.shape[1]:
        print(f"⏩ HVG: n_hvg={n_hvg}, using all {x.shape[1]} genes.")
        hvg_indices = np.arange(x.shape[1])
    else:
        print(f"🧬 Selecting top {n_hvg} highly variable genes by variance...")
        gene_var = x.var(axis=0)
        hvg_indices = np.argsort(-gene_var)[:n_hvg]
        print(f"✅ Selected {len(hvg_indices)} genes.")
    x_hvg = x[:, hvg_indices]
    print(f"✂️  x reduced from {x.shape} → {x_hvg.shape}")
    x_hvg_path = cache_dir / "x_hvg.npy"
    np.save(x_hvg_path, x_hvg)
    print(f"💾 Saved to {x_hvg_path}")
    # Update meta.json
    meta_path = cache_dir / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    meta["hvg"] = {
        "n_hvg_requested": n_hvg,
        "n_hvg_actual": len(hvg_indices),
        "method": "variance",
        "gene_indices_kept": hvg_indices.tolist(),
        "original_x_shape": list(x.shape),
        "new_x_shape": list(x_hvg.shape),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Updated meta.json")
    return str(x_hvg_path)


class SimpleDataset:
    def __init__(self, x, y, missing_value=-1.0):
        self.x = x
        self.y = y
        self.missing_value = missing_value  

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        unknown_mask = (y == self.missing_value)
        return x, y, unknown_mask   