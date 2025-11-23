# config/factory.py
from typing import List, Dict, Any, Iterator, Tuple, Optional
from itertools import product
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from .config_base import TrainConfig, OptimizerConfig, BatchScheduleConfig, LossConfig, ModelConfig
from itertools import product
from copy import deepcopy
from typing import Optional, List, Any, Dict, Iterator, Tuple
import re


def make_default_config() -> TrainConfig:
    """创建默认配置实例"""
    return TrainConfig()

def update_config(config: TrainConfig, updates: Dict[str, Any]) -> TrainConfig:
    """
    辅助的关键字更改函数
    cfg = update_config(cfg, {"optimizer.lr": 1e-3, "batch_schedule.subsample_max_samples": 5000})
    """
    config = deepcopy(config)
    for path, value in updates.items():
        parts = path.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return config

def _dict_to_run_name(param_dict: Dict[str, Any], max_len: int = 60) -> str:
    parts = []
    for k, v in sorted(param_dict.items()):
        k_short = k.replace("optimizer.", "opt_") \
                   .replace("batch_schedule.", "bs_") \
                   .replace("loss.", "loss_") \
                   .replace(".", "_")
        if isinstance(v, float):
            v_str = f"{v:.1e}".replace(".0", "")
        elif isinstance(v, bool):
            v_str = "T" if v else "F"
        else:
            v_str = str(v)
        parts.append(f"{k_short}={v_str}")
    name = "_".join(parts)
    return name[:max_len]

def grid_search_configs(
    seeds: Optional[List[int]] = None,
    base_config: Optional[TrainConfig] = None,
    **grids: List[Any]
) :
    if base_config is None:
        base_config = make_default_config()
    else:
        base_config = deepcopy(base_config)
    if seeds is None:
        seeds = [42]
    dotted_grids = {k.replace("__", "."): v for k, v in grids.items()}
    keys = list(dotted_grids.keys())
    for values in product(*dotted_grids.values()):
        param_dict = dict(zip(keys, values))
        cfg_template = update_config(base_config, param_dict)
        
        for seed in seeds:
            cfg = deepcopy(cfg_template)
            cfg.seed = seed
            parts = []
            for k, v in param_dict.items():
                clean_key = k.split('.')[-1]  
                if isinstance(v, float):
                    if abs(v) < 1e-4 or abs(v) >= 10:
                        v_str = f"{v:.1e}"   
                    else:
                        v_str = f"{v:.3g}"  
                else:
                    v_str = str(v)
                parts.append(f"{clean_key}={v_str}")
            parts.append(f"seed={seed}")
            run_name = "_".join(parts)
            run_name = re.sub(r"[^a-zA-Z0-9_.=-]", "_", run_name)  
            run_name = re.sub(r"_{2,}", "_", run_name)           
            
            yield param_dict, run_name, cfg, seed