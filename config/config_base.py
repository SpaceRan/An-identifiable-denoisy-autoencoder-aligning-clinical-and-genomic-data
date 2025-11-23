# config/base.py
from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class ModelConfig:
    """ICEBeeM 模型架构配置"""
    latent_dim: int = 192           # 公共隐空间维度 (d_z)
    f_monotonic: str = "decreasing"  # f: x → z, 必须为 "decreasing"
    g_monotonic: str = "increasing"  # g: y → z, 必须为 "increasing"

    hidden_activation: str = "leaky_relu"   # only "leaky_relu" allowed
    leaky_relu_slope: float = 0.2           # >0, satisfies assumption (a)
    noise_scale: float = 1e-3              # 前向加噪（防过拟合）,这个逻辑没有被用上
    
    # 先验（用于处理未知 y）
    prior_p: Optional[torch.Tensor] = None   # shape [cond_dim], 默认 0.5
    kappa: float = 5.0                       # Beta 分布集中参数


@dataclass
class OptimizerConfig:
    """优化器配置：统一 base_lr + schedule"""
    optimizer: str = "sgd"            
    base_lr: float = 0.05           # 主调节旋钮！

    # 学习率调度策略
    lr_schedule: str = "cosine"     # "cosine", "constant"
    warmup_steps_total: int = 500   # 改为 train_gpu.py 中变量名
    final_lr_ratio: float = 0.01    # cosine: final_lr = base_lr * ratio
    
    # Weight decay: 分组控制
    wd_wide_weights: float = 1e-4   # in_dim > 2000 的权重（防过拟合）
    wd_narrow_weights: float = 0.0  # in_dim ≤ 2000 的权重（防秩坍缩）
    wd_biases: float = 0.0
    wd_others: float = 0.0
    
    # SGD params，带动量
    momentum: float = 0.9
    nesterov: bool = True

    def __post_init__(self):
        assert self.base_lr > 0, "base_lr must be > 0"
        assert self.lr_schedule in ["cosine", "constant"], "Only 'cosine'/'constant' supported"
        assert self.warmup_steps_total >= 0, "warmup_steps_total >= 0"   
        if self.lr_schedule == "cosine":
            assert 0 < self.final_lr_ratio <= 1.0, "final_lr_ratio ∈ (0,1] for cosine"


@dataclass
class BatchScheduleConfig:
    """动态批大小配置panel"""
    B0: int = 64                 ### 第一个批次
    B_max: int = 1024            ### 最大批次
    batch_mult_every: int = 10   ### 多少步增加一下批次
    
    # 默认选择：子采样加速小 batch 训练
    subsample_when_small_bs: bool = True
    subsample_max_samples: int = 10000  # 小 batch 时数据集子采样上限

    def __post_init__(self):
        assert self.B0 > 0
        assert self.B_max >= self.B0
        assert self.batch_mult_every > 0


@dataclass
class LossConfig:
    temperature: float = 0.1         # InfoNCE 温度 τ；normalized 后推荐 0.05~0.2
    alpha_unknown: float = 0.5       # 未知样本损失权重（<1 表示削弱）    
    recon_weight: float = 0.5        # 重建损失的λ正则项, 需要网格搜索
    recon_loss_type: str = "mse"     # "mse" or "mae"
    start_early_stop_step = 1500     # 早停开始的最小step
    val_freq = 100                   # 早停:每多少步检测一下验证集损失,同时负责每多少步保存一次
    early_stop_min_delta =  5e-3     # 早停:最小改善精度
    early_stop_patience = 7          # 早停:最大容忍次数


    def __post_init__(self):
        assert self.temperature > 0, "temperature must be > 0"
        assert 0 <= self.alpha_unknown <= 1, "alpha_unknown ∈  [0,1]"
        assert self.recon_weight >= 0, "recon_weight ≥ 0"
        assert self.recon_loss_type in ["mse", "mae"], "recon_loss_type must be 'mse' or 'mae'"


class LossMonitor:
    """损失监控器
    忠实记录每一次的loss，并按照窗口设置返回平均的loss
    """
    def __init__(self, window_size: int = 50):
        assert window_size > 0, "window_size must be > 0"
        self.losses = []
        self.window = window_size
        self.history = []  # for plotting
    
    def update(self, loss: float):
        self.losses.append(loss)
        self.history.append(loss)
    
    @property
    def smoothed(self) -> float:
        """返回滑动平均，防小 batch 噪声"""
        if len(self.losses) < self.window:
            return float('nan')
        return float(sum(self.losses[-self.window:]) / self.window)
    

@dataclass
class TrainConfig:
    max_step: int = 7000            ## 单论训练中，最大训练次数为7000次
    seed: int = 42   
    val_ratio: float = 0.10             ##选择这么多作为验证集 
    missing_value: float = -1.0      # 缺失值标记
    hvg_genes: float = -1.0           # 是否筛选高变基因

    # 设备 & 数值
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True                 # 自动混合精度
    grad_clip: float = 1.0           # 梯度裁剪 norm
    num_workers: int = 4  
    
    # 子配置（组合式设计）
    model: ModelConfig = field(default_factory=ModelConfig)   # ← 加这行！   
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    batch_schedule: BatchScheduleConfig = field(default_factory=BatchScheduleConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # 保存 & 日志
    save_dir: str = "./model_results"
    log_rank_every: int = 100   
    def __post_init__(self):
        # 防止默认为 None
        if self.optimizer is None:
            self.optimizer = OptimizerConfig()
        if self.batch_schedule is None:
            self.batch_schedule = BatchScheduleConfig()
        if self.loss is None:
            self.loss = LossConfig()
