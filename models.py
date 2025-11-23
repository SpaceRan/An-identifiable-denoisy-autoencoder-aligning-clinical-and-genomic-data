import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from torch.distributions import Beta
import math
import math
from typing import List
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm   

def make_sn_linear(in_features, out_features, bias=True):
    lin = nn.Linear(in_features, out_features, bias=bias)
    spectral_norm(lin)                        
    return lin

def _golden_dims(input_dim: int, latent_dim: int, n_layers: int = 6) -> List[int]:
    if n_layers <= 0:
        return []
    if input_dim == latent_dim:
        return [input_dim] * n_layers
    input_dim = max(1, input_dim)
    latent_dim = max(1, latent_dim)
    log_a = math.log(input_dim)
    log_b = math.log(latent_dim)
    dims = []
    for i in range(n_layers):
        t = i / max(1, n_layers - 1) 
        log_val = log_a * (1 - t) + log_b * t
        val = math.exp(log_val)
        dim = int(round(val))
        low, high = min(input_dim, latent_dim), max(input_dim, latent_dim)
        dim = max(1, min(dim, high * 10)) 
        dims.append(dim)
    if input_dim > latent_dim:
        for i in range(1, len(dims)):
            dims[i] = min(dims[i], dims[i-1] - 1 if dims[i-1] > latent_dim else dims[i-1])
            if dims[i] < latent_dim:
                dims[i] = latent_dim
    else:
        for i in range(1, len(dims)):
            dims[i] = max(dims[i], dims[i-1] + 1 if dims[i-1] < latent_dim else dims[i-1])
            if dims[i] > latent_dim:
                dims[i] = latent_dim
    while len(dims) < n_layers:
        dims.append(latent_dim)
    return dims[:n_layers]

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 196, n_layers: int = 6,
                 non_monotonic: bool = True):          # ← 新增开关
        super().__init__()
        self.non_monotonic = non_monotonic
        self.base_latent_dim = latent_dim        
        final_latent_dim = latent_dim * 3 if non_monotonic else latent_dim
        hidden_dims = _golden_dims(input_dim, final_latent_dim, n_layers)
        layers = []
        layers.append(make_sn_linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(len(hidden_dims) - 1):
            layers.append(make_sn_linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(make_sn_linear(hidden_dims[-1], self.base_latent_dim))
        self.backbone = nn.Sequential(*layers)        
        self.register_buffer("dummy", torch.tensor(0.0))  
        
        dims_str = " → ".join(f"{d:4d}" for d in hidden_dims)
        print(f"Encoder: {input_dim:4d} → {dims_str} → {self.base_latent_dim}"
              f"{' → [z,z²,0.1z³] (×3)' if non_monotonic else ''}")
    
    def forward(self, x):
        z = self.backbone(x)                        
        if not self.non_monotonic:
            return z
        z1 = z
        z2 = z ** 2
        z3 = 0.1 * (z ** 3)
        z_aug = torch.cat([z1, z2, z3], dim=1)                
        return z_aug, z                                     
    
class ConditionEncoder(nn.Module):
    def __init__(self, cond_dim: int, latent_dim: int = 196, n_layers: int = 6):
        super().__init__()
        self.base_dim = latent_dim
        hidden_dims = _golden_dims(cond_dim, latent_dim, n_layers)  # ← output to base_dim
        layers = []
        layers.append(make_sn_linear(cond_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(len(hidden_dims)-1):
            layers.append(make_sn_linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(make_sn_linear(hidden_dims[-1], latent_dim))  # ← to base_dim
        self.backbone = nn.Sequential(*layers)        
        print(f"ConditionEncoder (clinical): {cond_dim} → ... → {latent_dim} → [z,z²,0.1z³] (×3)")

    def forward(self, x):
        z = self.backbone(x)                     # (B, latent_dim)
        z1 = z
        z2 = z ** 2
        z3 = 0.1 * (z ** 3)
        z_aug = torch.cat([z1, z2, z3], dim=1)    # (B, 3 * latent_dim)
        return z_aug, z                          # ← consistent with Encoder!

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 196, output_dim: int = 20000, n_layers: int = 6):
        super().__init__()
        input_dim = latent_dim * 3
        hidden_dims = _golden_dims(input_dim, output_dim, n_layers)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self._init_linear(layers[-1])
        layers.append(nn.LeakyReLU(0.2))
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))  # ← plain Linear
            self._init_linear(layers[-1])
            layers.append(nn.LeakyReLU(0.2))        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self._init_linear(layers[-1])
        self.net = nn.Sequential(*layers)
        dims_str = " → ".join(f"{d}" for d in hidden_dims)
        print(f"Decoder : {input_dim} → {dims_str} → {output_dim}")
    def _init_linear(self, layer: nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, a=0.2, nonlinearity='leaky_relu')
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(layer.bias, -bound, bound)
    def forward(self, x):
        noise = torch.randn_like(x) * 0.01
        x = x + noise
        return self.net(x)
    
    
class ICEBeeM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int = 128,
        noise_scale: float = 1e-3,
        prior_p: Optional[torch.Tensor] = None,
        kappa: float = 5.0,
        missing_value: float = -1.0,
        recon_weight: float = 0.1  
    ):
        super().__init__()
        self.f = Encoder(input_dim, latent_dim=latent_dim, non_monotonic=True)
        self.g = ConditionEncoder(cond_dim, latent_dim=latent_dim)
        self.h = Decoder(latent_dim=latent_dim, output_dim=input_dim)
        self.latent_dim = latent_dim * 3             
        self.noise_scale = noise_scale
        self.kappa = kappa
        self.missing_value = missing_value
        self.recon_weight = recon_weight  # 用于平衡 NCE 和重建损失

        prior_p = prior_p if prior_p is not None else torch.full((cond_dim,), 0.5)
        self.register_buffer('prior_buffer', prior_p)


    def forward(
        self,
        x: torch.Tensor,        
        y: torch.Tensor,
        training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_clean = x.clone().detach()
        if training and self.noise_scale > 0:
            std_x = x.std(dim=0, keepdim=True).clamp(min=1e-6)
            noise = torch.randn_like(x) * (self.noise_scale * std_x)
            x_noisy = x + noise
        else:
            x_noisy = x

        y_processed = self._process_condition(y, training)    
        z_x_aug, z_x_raw = self.f(x_noisy)          # (B, 588), (B, 198)  ← assuming latent_dim=198
        z_y_aug, z_y_raw = self.g(y_processed)      # ← FIX: unpack both!
        x_hat = self.h(z_x_aug)                     # decoder uses augmented (588-d)
        return z_x_aug, z_y_aug, x_hat, x_clean, z_x_raw, z_y_raw  

    def _process_condition(self, y: torch.Tensor, training: bool) -> torch.Tensor:
        y_processed = y.clone()
        for j in range(y.shape[1]):
            mask = y[:, j] == self.missing_value
            if mask.any():
                p_j = self.prior_buffer[j]
                if training:
                    alpha = max(p_j * self.kappa, 1e-4)
                    beta = max((1.0 - p_j) * self.kappa, 1e-4)
                    alpha = alpha.to(y.device)
                    beta = beta.to(y.device)
                    sampled = Beta(alpha, beta).sample([mask.sum()]).to(y.dtype)
                    y_processed[mask, j] = sampled
                else:
                    y_processed[mask, j] = p_j
        return y_processed


def nce_loss(
    z_x_raw: torch.Tensor,   # (B, d)
    z_y_raw: torch.Tensor,   # (B, d) ← now use raw y, not aug
    temperature: float = 0.5,
    alpha_unknown: float = 0.5,
    unknown_mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    B, d = z_x_raw.shape
    assert z_y_raw.shape == (B, d), f"Shape mismatch: {z_x_raw.shape} vs {z_y_raw.shape}"
    logits = torch.matmul(z_x_raw, z_y_raw.t()) / temperature  # ← raw-to-raw!
    logits = logits.clamp(min=-50, max=50)
    labels = torch.arange(B, device=logits.device)
    loss = F.cross_entropy(logits, labels, reduction='none')
    if unknown_mask is not None and unknown_mask.any():
        weights = torch.ones(B, device=loss.device)
        weights[unknown_mask.any(dim=1)] = alpha_unknown
        loss = (loss * weights).sum() / (weights.sum() + 1e-8)
    else:
        loss = loss.mean()

    return loss