# main.py
import os
import multiprocessing as mp
import torch
import sys
import numpy as np
from pathlib import Path
import gc
import argparse
import pandas as pd
import json
from dataset import preprocess_paired_csv_to_npy, load_dataset_from_npy, apply_hvg_filter
from train import train_one_run
from config import grid_search_configs, make_default_config
from config.factory import update_config
import sys
from datetime import datetime
from pathlib import Path

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, txt):
        for f in self.files:
            f.write(txt)
            f.flush()  # 实时写入，方便 tail -f

    def flush(self):
        for f in self.files:
            f.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="ICEBeeM Training")
    parser.add_argument("--gene-csv", type=str, help="Path to gene CSV")
    parser.add_argument("--clinical-csv", type=str, help="Path to clinical CSV")
    parser.add_argument("--id-col", type=str, default="ID")
    parser.add_argument("--target-cols", type=str, nargs="+", help="y columns")
    parser.add_argument("--x-path", type=str, default="./src/x.npy",
                        help="Path to x.npy (relative to current working directory)")
    parser.add_argument("--y-path", type=str, default="./src/y.npy",
                        help="Path to y.npy")
    parser.add_argument("--reprocess", action="store_true",
                        help="Force re-run CSV → npy")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Override steps for quick test (<=0: use default)")
    parser.add_argument("--grid-seeds", type=int, nargs="+", default=[42],
                        help="Seeds for grid search")
    parser.add_argument("--hvg-genes",type=int,default=-1.0,
                        help="Number of highly variable genes to select (default: all). Set <=0 to use all genes.") 
                        ## 这个超参数暂时无法网格搜索，但是我觉得就这样也可以
    args = parser.parse_args()
    x_path = Path(args.x_path)
    y_path = Path(args.y_path)
    if not args.clinical_csv and not (x_path.exists() and y_path.exists()):
        parser.error(
            "Either provide  --gene-csv &--clinical-csv & --target-cols,\n"
            "or ensure x.npy & y.npy exist at --x-path / --y-path.")
    return args


def main():
    args = parse_args()
    x_path = Path(args.x_path)
    y_path = Path(args.y_path)
    need_preprocess = (
        args.reprocess
        or not x_path.exists()
        or not y_path.exists()
    )
    if need_preprocess:
        if not (args.clinical_csv and args.gene_csv and args.target_cols):
            raise ValueError(
                "Need --gene-csv, --clinical-csv, --target-cols to preprocess."
            )
        cache_dir = x_path.parent
        print(f"🔄 Preprocessing CSVs → {cache_dir}")
        preprocess_paired_csv_to_npy(
            clinical_csv=args.clinical_csv,
            gene_csv=args.gene_csv,
            id_col=args.id_col,
            target_cols=args.target_cols,
            cache_dir=str(cache_dir),
        )
        print("✅ npy files generated.")
    original_gene_count = np.load(x_path).shape[1]  # Get the actual number of genes in x.npy
    
    if args.hvg_genes != original_gene_count and args.hvg_genes > 0:
        cache_dir = x_path.parent
        print(f"🔬 Current gene count: {original_gene_count}, requested HVG: {args.hvg_genes}")
        print(f"🔬 Applying HVG filter: selecting {args.hvg_genes} highly variable genes")
        x_path = apply_hvg_filter(
            x_path=str(x_path),
            cache_dir=str(cache_dir),
            n_hvg=args.hvg_genes,
        )
        print(f"✅ HVG filtering completed. New x path: {x_path}")
    elif args.hvg_genes <= 0:
        print(f"⏭️ HVG genes parameter is <= 0, using all {original_gene_count} genes")
    else:
        print(f"⏭️ Requested HVG genes ({args.hvg_genes}) matches current gene count ({original_gene_count}), skipping HVG filtering")

    base_config = make_default_config()
    if args.max_steps > 0:
        base_config = update_config(base_config, {"steps": args.max_steps})


    # ─── 2️⃣ ✅ 关键：所有日志基于 cfg.save_dir ───
    save_dir = Path(base_config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 全局日志：save_dir/YYYYMMDD_HHMMSS_main.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_log_path = save_dir / f"{timestamp}_main.log"
    global_log_file = open(global_log_path, "w", encoding="utf-8", buffering=1)

    # 重定向 stdout/stderr（终端 + 全局日志双写）
    sys.stdout = Tee(sys.stdout, global_log_file)
    sys.stderr = Tee(sys.stderr, global_log_file)
    
    # 记录所有参数和配置信息
    print(f"[LOG] Global log: {global_log_path}")
    print(f"[CONFIG] Arguments: {args}")
    print(f"[CONFIG] Original gene count: {original_gene_count}")
    print(f"[CONFIG] HVG genes requested: {args.hvg_genes}")
    print(f"[CONFIG] Final x path: {x_path}")
    print(f"[CONFIG] Base config: {base_config}")
    
    # ─── 3️⃣ 开始 grid search ───
    all_results = []
    total_runs = 0
    for param_dict, run_name, cfg, seed in grid_search_configs(
        base_config=base_config,
        seeds=[42],
        optimizer__base_lr=[0.05, 0.1],
        loss__temperature=[0.05, 0.1],
        loss__recon_weight=[0.5],
        loss__alpha_unknown=[0.2, 0.4],
    ):
        total_runs += 1
        print(f"\n{'=' * 50}")
        print(f"🚀 Run {total_runs}: {run_name}")
        print(f"{'=' * 50}")
        # 每个 run 的独立目录（save_dir / run_name）
        run_dir = save_dir / run_name
        run_dir.mkdir(exist_ok=True)
        # ✅ 单 run 日志：run_dir/YYYYMMDD_HHMMSS_run.log
        run_log_path = run_dir / f"{timestamp}_run.log"
        run_log_file = open(run_log_path, "w", encoding="utf-8", buffering=1)
        print(f"[LOG] Run log: {run_log_path}")
        # 临时重定向：terminal + global_log + run_log
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, global_log_file, run_log_file)
        sys.stderr = Tee(sys.stderr, global_log_file, run_log_file)

        try:
            print("📦 Loading dataset...")
            full_dataset, val_ds = load_dataset_from_npy(
                x_path=str(x_path), 
                y_path=str(y_path),
                val_ratio=getattr(cfg, "val_ratio"),
                missing_value=getattr(cfg, "missing_value"),
                seed=getattr(cfg, "seed"),
            )
            print(f"✅ Dataset loaded: x={full_dataset.x.shape}, y={full_dataset.y.shape}")
            metrics = train_one_run(
                cfg=cfg,
                dataset_train=full_dataset,
                dataset_val=val_ds,
                run_dir=run_dir,
            )

            print(f"Now Model config: {cfg.model}")
            print(f"Run {run_name} metrics: {metrics}")
            metrics.update({"run_name": run_name, "seed": seed})
            all_results.append(metrics)

        except Exception as e:
            import traceback
            print(f"❌ {run_name}: {e}")
            traceback.print_exc()
        finally:
            # 恢复 stdout/stderr + 关闭 run log
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            run_log_file.close()

    # ─── 总结 ───
    summary_path = save_dir / "grid_search_summary.csv"
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(summary_path, index=False)
        print(f"\n✅ Summary saved to {summary_path}")
        print(f"📊 Summary: {len(all_results)} successful runs out of {total_runs}")
        for result in all_results:
            print(f"   - {result.get('run_name', 'Unknown')}: {result}")
    else:
        print("⚠️ No successful runs.")

    print(f"\n🔚 Grid search done. {len(all_results)}/{total_runs} succeeded.")
    global_log_file.close()  # 关闭全局日志

if __name__ == '__main__':
    main()
