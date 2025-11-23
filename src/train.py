# src/train_gpu.py
import math
import time
import random
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda import amp
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from torch.cuda.amp import GradScaler
from models import ICEBeeM, nce_loss

# ==================总体内容相关================
@dataclass
class TrainerState:
    global_step: int = 0
    best_train_nce: float = float('inf')
    best_train_recon: float = float('inf')
    best_train_total: float = float('inf')
    best_val_nce: float = float('inf')
    best_val_recon: float = float('inf')
    best_val_total: float = float('inf')
    steps_no_improve: int = 0   
    overfitting_count: int = 0  
    early_stop_triggered: bool = False
## 前置状态声明


def setup_device_and_seed(cfg) -> torch.device:
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # —— 设备设置 ——
    device = torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("GPU required for training.")
    print(f"✅ Seed={seed} set | Device: {device} | GPU: {torch.cuda.get_device_name()}")
    return device



## =====================全新的样本loder机制
class FullBatchProvider:
    """
    一次性加载整个 dataset (X, Y, Mask) 到 CPU 内存中，
    然后根据 step_to_batch 动态返回对应 batch。
    """
    def __init__(self, dataset, step_to_batch, subsample_indices=None, device="cpu"):
        self.device = device
        self.step_to_batch = step_to_batch
        self.subsample_indices = subsample_indices or {}
        # === 加载整个 dataset 到 CPU ===
        X_list, Y_list, M_list = [], [], []
        for x, y, m in DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=4):
            X_list.append(x)
            Y_list.append(y)
            M_list.append(m)
        self.X_all = torch.cat(X_list, dim=0)
        self.Y_all = torch.cat(Y_list, dim=0)
        self.M_all = torch.cat(M_list, dim=0)
        self.N = len(self.X_all)
        self.indices = torch.arange(self.N)
        self.pos = 0  # 当前 epoch 位置

        self.shuffle()
    def shuffle(self):
        perm = torch.randperm(self.N)
        self.indices = self.indices[perm]
        self.pos = 0
    def next_batch(self, step):
        bs = self.step_to_batch[step]
        # 若启用 subsample_indices
        if step in self.subsample_indices:
            idx = torch.tensor(self.subsample_indices[step])
            idx = idx[torch.randperm(len(idx))[:bs]]
            return (
                self.X_all[idx].to(self.device, non_blocking=True),
                self.Y_all[idx].to(self.device, non_blocking=True),
                self.M_all[idx].to(self.device, non_blocking=True)
            )
        if self.pos + bs > self.N:
            self.shuffle()
        idx = self.indices[self.pos:self.pos + bs]
        self.pos += bs
        return (
            self.X_all[idx].to(self.device, non_blocking=True),
            self.Y_all[idx].to(self.device, non_blocking=True),
            self.M_all[idx].to(self.device, non_blocking=True)
        )
    

# ======================== 状态与断点续传机制
def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint (strict format: current version only).
    Must contain: model, optimizer, scheduler, scaler (for AMP), training state.
    """
    if not checkpoint_path.exists():
        print(f"📁 No checkpoint found at {checkpoint_path}")
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"🔁 Resuming from {checkpoint_path}")
        resume_dict = {
            'model_state_dict': ckpt['model_state_dict'],
            'optimizer_state_dict': ckpt['optimizer_state_dict'],
            'scheduler_state_dict': ckpt['scheduler_state_dict'],
            'scaler_state_dict': ckpt['scaler_state_dict'],   
            'global_step': ckpt['global_step'],
            'best_train_nce': ckpt['best_train_nce'],
            'best_train_recon': ckpt['best_train_recon'],
            'best_train_total': ckpt['best_train_total'],
            'best_val_nce': ckpt['best_val_nce'],
            'best_val_recon': ckpt['best_val_recon'],
            'best_val_total': ckpt['best_val_total'],
            'steps_no_improve': ckpt['steps_no_improve'],
            'overfitting_count': ckpt['overfitting_count'],
            'early_stop_triggered': ckpt['early_stop_triggered'],
            'config': ckpt.get('config', {}),  
        }
        print(f"   → Step {resume_dict['global_step']}, "
              f"Best Val NCE: {resume_dict['best_val_nce']:.4f}")
        return resume_dict
    except KeyError as e:
        print(f"❌ Missing key in checkpoint: {e}. "
              "→ Checkpoint format outdated. Please retrain or manually update.")
        return None
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None

def save_checkpoint(
    run_dir: Path,
    model, optimizer, scheduler, scaler,  
    cfg, state: TrainerState,            
    is_best: bool = False):

    checkpoint_path = run_dir / "model.pt"
    best_path = run_dir / "model_best.pt"
    checkpoint = {
        'global_step': state.global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),  
        'config': asdict(cfg) if hasattr(cfg, '__dataclass_fields__') else cfg.__dict__,
        'best_train_nce': state.best_train_nce,
        'best_train_recon': state.best_train_recon,
        'best_train_total': state.best_train_total,
        'best_val_nce': state.best_val_nce,
        'best_val_recon': state.best_val_recon,
        'best_val_total': state.best_val_total,
        'steps_no_improve': state.steps_no_improve,
        'overfitting_count': state.overfitting_count,
        'early_stop_triggered': state.early_stop_triggered,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Saved checkpoint (step {state.global_step}) to {checkpoint_path}")
    
    if is_best:
        torch.save(model.state_dict(), best_path)  # only weights for best
        print(f"⭐ Saved best model (val NCE={state.best_val_nce:.4f}) to {best_path}")

# ======================== 模型的基本元素构造
def build_model_optimizer_total_steps(cfg, dataset, device):
    """
    构建 model, optimizer,并返回total_steps
    """
    # === Model ===========================================================
    # 我们首先就在这里设置了一下input_dim 和 cond_dim
    input_dim = dataset.x.shape[1] if hasattr(dataset, 'x') else dataset.x_dim
    cond_dim = dataset.y.shape[1] if hasattr(dataset, 'y') else dataset.y_dim
    model_cfg = cfg.model
    model = ICEBeeM(
        input_dim=input_dim,
        cond_dim=cond_dim,
        latent_dim=model_cfg.latent_dim,
        noise_scale=model_cfg.noise_scale,
        kappa=model_cfg.kappa,
        missing_value=cfg.missing_value 
            ).to(device)
    # === Optimizer (SGD groups) ============================================
    named_params = dict(model.named_parameters())
    weights = [p for n, p in named_params.items() if 'weight' in n]
    biases = [p for n, p in named_params.items() if 'bias' in n]
    others = [p for n, p in named_params.items() 
              if ('weight' not in n) and ('bias' not in n)]
    wide_weights = [w for w in weights if w.shape[1] > 2000]
    narrow_weights = [w for w in weights if w.shape[1] <= 2000]
    opt_cfg = cfg.optimizer
    base_lr = opt_cfg.base_lr
    param_groups = [
        {'params': wide_weights,   'lr': base_lr, 'weight_decay': opt_cfg.wd_wide_weights},
        {'params': narrow_weights, 'lr': base_lr, 'weight_decay': opt_cfg.wd_narrow_weights},
        {'params': biases,         'lr': base_lr, 'weight_decay': opt_cfg.wd_biases},
        {'params': others,         'lr': base_lr, 'weight_decay': opt_cfg.wd_others}
    ]
    optimizer = SGD(
        param_groups,
        momentum=opt_cfg.momentum,
        nesterov=opt_cfg.nesterov
    )

    total_steps = cfg .max_step
    print(f"📊 Max total steps: {total_steps}")
    
    # === AMP Scaler ============没什么意义，因为AMD加速我认为是没有什么可讨论的
    scaler = None
    if cfg.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        print("✅ AMP enabled: GradScaler initialized")
    return model, optimizer , scaler, total_steps

# ==================================== 数据计划设计
def prepare_batch_schedule(
    cfg,
    dataset,
    start_step: int = 0):
    """
    预计算每个 step 的 batch size 和 subsample 索引
    Returns:
        step_to_batch: Dict[step, batch_size]
        subsample_indices: Dict[step, indices]  # 小 batch 时的子采样
    """
    bs_cfg = cfg.batch_schedule
    B0 = bs_cfg.B0
    B_max = bs_cfg.B_max
    batch_mult_every = bs_cfg.batch_mult_every  
    # === 1. 计算每个 step 的 batch size ===
    step_to_batch = {}
    current_bs = B0
    steps_since_mult = 0
    # 预计算所有 steps 的 batch size
    for step in range(start_step, cfg.max_step):
        step_to_batch[step] = current_bs
        steps_since_mult += 1
        if steps_since_mult >= batch_mult_every and current_bs < B_max:
            current_bs = min(current_bs * 2, B_max)
            steps_since_mult = 0

    # === 2. 计算 subsample indices (per step) ===
    subsample_indices: Dict[int, np.ndarray] = {}
    if getattr(bs_cfg, 'subsample_when_small_bs', False):
        threshold = getattr(bs_cfg, 'subsample_bs_threshold', 32)
        max_samples = getattr(bs_cfg, 'subsample_max_samples', len(dataset))
        for step in range(start_step, cfg.max_step):
            bs = step_to_batch[step]
            if bs < threshold and len(dataset) > max_samples:
                # 每 step 独立 RNG (seed + step 保证可复现)
                step_rng = np.random.default_rng(seed=cfg.seed + step)
                indices = step_rng.choice(
                    len(dataset),
                    size=max_samples,
                    replace=False
                )
                subsample_indices[step] = indices
    return step_to_batch, subsample_indices

# scheduler=========================
def create_step_based_scheduler(
    optimizer,
    warmup_steps: int = 500,
    total_steps: int = 10000,
    final_lr_ratio: float = 0.01
):
    """
    创建 step-based 学习率调度器
    optimizer: 优化器实例
    warmup_steps: 预热步数（前 warmup_steps 步线性增加）
    total_steps: 总训练步数（用于 cosine decay）
    final_lr_ratio: 最终学习率比例（final_lr = base_lr * ratio）
    Returns:
        LambdaLR 调度器实例
    """
    def lr_lambda(step: int): ## 自动维护计数器，只需激活scheduler.step()即可
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = max(0.0, min(1.0, progress))
            return final_lr_ratio + (1 - final_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==================================== 按照计划进行数据准备
# ====================================  这个现在只有val在用
def create_dataloader(
    dataset,          
    batch_size: int,
    num_workers: int = 4,  
    shuffle: bool = True,         
    drop_last: bool = True,      
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,         
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

# =================================== 有效秩检验，逻辑简单不用复查 ==============
# ------------------------------由于实在是太大了，因此我们实际上没有用下面的的函数----------------------
def effective_rank(matrix: torch.Tensor) -> float:
    if matrix.ndim != 2 or matrix.numel() == 0:
        return float('nan')
    s = torch.linalg.svdvals(matrix)
    if s.sum().abs() < 1e-12:
        return 0.0
    p = s / s.sum()
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    return math.exp(entropy.item())


def get_ranks(model) :
    ranks = {}
    for name, param in model.named_parameters():
        if param.ndim != 2: 
            continue
        eff_rank = effective_rank(param.data)
        max_rank = min(param.shape)
        ranks[name] = (eff_rank, max_rank)
    return ranks

def print_ranks(ranks_dict,step ):
    """打印秩信息"""
    print(f"\n[Step {step}] Effective Ranks (2D params only):")
    for name, (eff_rank, max_rank) in ranks_dict.items():
        print(f"  {name:25} | eff_rank: {eff_rank:5.1f} / {max_rank}")


#============================一次训练中真正的停止机制 
def compute_batch_loss(
    model,
    batch,
    device,
    cfg,
    training: bool = True
):
    """通用 loss 计算函数"""
    x, y, unknown_mask = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    unknown_mask = unknown_mask.to(device, non_blocking=True)
    z_x_aug, z_y_aug, x_hat, x_clean, z_x_raw, z_y_raw = model(x, y, training=training)    
    z_x = F.normalize(z_x_raw, dim=-1, p=2)   
    z_y = F.normalize(z_y_raw, dim=-1, p=2) 
    contrastive_loss = nce_loss(
        z_x, z_y,                                    # z_x_raw, z_y_raw
        cfg.loss.temperature,                        # temperature (位置3)
        cfg.loss.alpha_unknown,                      # alpha_unknown (位置4)  
        unknown_mask                                 # unknown_mask (位置5)
    ) 
    recon_loss = F.mse_loss(x_hat, x_clean)
    total_loss = contrastive_loss + cfg.loss.recon_weight * recon_loss
    
    return contrastive_loss, recon_loss, total_loss

## 一次梯度传播

def train_one_step(
    model,
    optimizer,
    scheduler,
    val_loader,
    device,
    cfg,
    state: TrainerState,
    current_step: int,
    batch,
    scaler=None,
    run_dir: Optional[Path] = None,
    skip_first_val_save: bool = False, 
):
    step_start = time.time()
    if current_step == 0:
        state.train_start_time = time.time()

    model.train()
    
    contrastive_loss, recon_loss, train_loss = compute_batch_loss(
        model, batch, device, cfg, training=True
    )
    
    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(train_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()
    scheduler.step()

    step_time = time.time() - step_start
    should_print_speed = (current_step < 100 or 
        (current_step >= 100 and current_step % 100 < 5) )
    
    if should_print_speed:
        print(f"⏱️ Step {current_step+1} | Step time: {step_time:.4f}s")
            
    start_early_stop_step = getattr(cfg.loss, 'start_early_stop_step', 2000)
    val_freq = getattr(cfg.loss, 'val_freq', 100)

    if current_step < start_early_stop_step:  ## 早停制止逻辑
        if current_step % val_freq == 0:
            print(f"🔍 Step {current_step:6d} | "
                f"Train: {train_loss.item():.6f} | "
                f"Train(NCE:{contrastive_loss.item():.4f},Recon:{recon_loss.item():.4f})")
        return False, state, train_loss.item()

    min_delta = getattr(cfg.loss, 'early_stop_min_delta', 5e-4)
    if current_step % val_freq != 0:
        return False, state, train_loss.item()
        
    # --- 验证逻辑 ---
    model.eval()
    val_contrastive = val_recon = 0.0
    val_total = 0.0
    val_count = 0
    with torch.no_grad():
        for val_batch in val_loader:
            contrastive, recon, total = compute_batch_loss(
                model, val_batch, device, cfg, training=False
            )
            val_contrastive += contrastive.item()
            val_recon += recon.item()
            val_total += total.item()
            val_count += 1
            
    avg_val_contrastive = val_contrastive / val_count
    avg_val_recon = val_recon / val_count
    avg_val_total = val_total / val_count
    
    print(f"🔍 Step {current_step:6d} | "
          f"Train: {train_loss.item():.6f} | Val: {avg_val_total:.6f} | "
          f"Train(NCE:{contrastive_loss.item():.4f},Recon:{recon_loss.item():.4f}) | "
          f"Val(NCE:{avg_val_contrastive:.4f},Recon:{avg_val_recon:.4f})")
    
    val_improved = (state.best_val_total - avg_val_total) > min_delta    
    if val_improved:
        state.best_val_total = avg_val_total
        state.best_val_nce = avg_val_contrastive
        state.best_val_recon = avg_val_recon
        state.steps_no_improve = 0
        state.best_train_total = min(state.best_train_total, train_loss.item())
        state.best_train_nce = min(state.best_train_nce, contrastive_loss.item())
        state.best_train_recon = min(state.best_train_recon, recon_loss.item())
        if current_step > 0 and run_dir and not skip_first_val_save:
            save_checkpoint(run_dir, model, optimizer, scheduler, scaler, cfg, state, is_best=True)
            print(f"⭐ Saved best checkpoint (Train: {train_loss.item():.6f}, Val: {avg_val_total:.6f})")
    else:
        state.steps_no_improve += 1
        print(f"⚠️ Val not improved | "
              f"Val Δ: {avg_val_total - state.best_val_total:+.6f} | "
              f"No-improve time: {state.steps_no_improve}")
              
    patience = getattr(cfg, 'early_stop_patience', 10)
    if state.steps_no_improve >= patience:
        print(f"\n🛑 Early stopping at step {current_step} | "
              f"Best Val: {state.best_val_total:.6f} (Train: {state.best_train_total:.6f})")
        state.early_stop_triggered = True
        return True, state, train_loss.item()
        
    return False, state, train_loss.item()

# ====整体训练======================================#
def train_one_run(
    cfg,
    dataset_train,
    dataset_val=None,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    import time
    start_time = time.time()
    device = setup_device_and_seed(cfg)

    # === 1. Setup ===
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "model.pt" 
    state = TrainerState()                 # ← initial state
    # === 2. Resume or init ===
    ckpt_data = None
    skip_first_val_save_local = False
    if checkpoint_path and checkpoint_path.exists():
        ckpt_data = load_checkpoint(checkpoint_path)  # 进行良好的数据提取
        if ckpt_data:
            state.global_step = ckpt_data.get('global_step', 0)
            state.best_train_nce = ckpt_data.get('best_train_nce', float('inf'))
            state.best_train_recon = ckpt_data.get('best_train_recon', float('inf'))
            state.best_train_total = ckpt_data.get('best_train_total', float('inf'))
            state.best_val_nce = ckpt_data.get('best_val_nce', float('inf'))
            state.best_val_recon = ckpt_data.get('best_val_recon', float('inf'))
            state.best_val_total = ckpt_data.get('best_val_total', float('inf'))
            state.steps_no_improve = ckpt_data.get('steps_no_improve', 0)
            state.overfitting_count = ckpt_data.get('overfitting_count', 0)
            state.early_stop_triggered = ckpt_data.get('early_stop_triggered', False)
            print("❤️checkpoint loaded.")
            skip_first_val_save_local = True
            if state.early_stop_triggered:
                print("⚠️ Warning: checkpoint was early-stopped. Training will resume, "
                      "but may stop immediately again.")
    done_file = run_dir / "done.txt"
    
    if done_file.exists():
        print(f" ⏭️ ❤️Found {done_file}, skipping training as requested.")
        return {
            "final_loss": ckpt_data.get('best_val_total', float('inf')) if ckpt_data else float('inf'),
            "early_stopped": True,
            "stopped_at_step": ckpt_data.get('global_step', 0) if ckpt_data else 0,
            "total_time_min": 0.0,
        }
    # === 3. Build model, optimizer, etc. ===
    device = torch.device(cfg.device if hasattr(cfg, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer, scaler, total_steps = build_model_optimizer_total_steps(cfg, dataset_train, device)

    # === 4. Load checkpoint weights ===
    if ckpt_data:
        model.load_state_dict(ckpt_data['model_state_dict'])
        optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt_data and scaler is not None:
            scaler.load_state_dict(ckpt_data['scaler_state_dict'])

    # === 5. Create schedulers and data loaders ===
    scheduler = create_step_based_scheduler(
        optimizer,
        warmup_steps=getattr(cfg, 'warmup_steps', 500),
        total_steps=total_steps,
        final_lr_ratio=getattr(cfg, 'final_lr_ratio', 0.01)
    )
    if ckpt_data and 'scheduler_state_dict' in ckpt_data:
        scheduler.load_state_dict(ckpt_data['scheduler_state_dict'])
    
    # Prepare batch schedule
    step_to_batch, subsample_indices = prepare_batch_schedule(
        cfg, dataset_train, start_step=state.global_step
    )

    ## 重铸dataloder，全部加载到内存
    provider = FullBatchProvider(
        dataset_train,
        step_to_batch,
        subsample_indices=subsample_indices,
        device=device
    )
    val_loader = create_dataloader(
        dataset=dataset_val,
        batch_size=getattr(cfg, 'val_batch_size', 64), 
        shuffle=False,
        drop_last=False,
        num_workers=getattr(cfg, 'num_workers', 4),
    ) if dataset_val is not None else None
    if val_loader is None:
        print("⚠️ Warning: No validation dataset provided. Training without validation.")
        
    # === 6. Training loop ===
    train_loss = float('inf') 
    for current_step in range(state.global_step, cfg.max_step):
        x_batch, y_batch, um_batch = provider.next_batch(current_step)
        batch = (x_batch, y_batch, um_batch)
        
        early_stop, state, train_loss = train_one_step(
                model, optimizer, scheduler, val_loader, device, cfg, state,
                current_step, batch, scaler, run_dir,
                skip_first_val_save=skip_first_val_save_local
            )
        skip_first_val_save_local = False
        state.global_step += 1
        if early_stop:
            break

        # Save checkpoint periodically 
        # if run_dir and current_step % getattr(cfg, 'save_freq', 1000) == 0:
        #    save_checkpoint(run_dir, model, optimizer, scheduler, scaler, cfg, state, is_best=False)

        # Final save
    if run_dir:
        save_checkpoint(run_dir, model, optimizer, scheduler, scaler, cfg, state, is_best=False)

    #final_step = state.global_step
    #final_ranks = get_ranks(model) 
    #print_ranks(final_ranks, final_step)
    ### 这个估计要算超级久，就直接算了

    return {
        "final_loss": train_loss,
        "early_stopped": state.early_stop_triggered,
        "stopped_at_step": state.global_step,
        "total_time_min": (time.time() - start_time) / 60,
    }