import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBaseline(nn.Module):
    """
   Baseline без графовых сверток. 
    Проверяет, достаточно ли информации в сырых признаках (1028D и 18D) 
    для сопоставления 2D и 3D узлов.
    """
    def __init__(self, input_dim_2d: int = 1028, input_dim_3d: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.encoder_2d = nn.Sequential(
            nn.Linear(input_dim_2d, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.encoder_3d = nn.Sequential(
            nn.Linear(input_dim_3d, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )

    def forward(self, x_dict, edge_index_dict):
        # Полностью игнорируем edge_index_dict (топологию)
        z_2d = self.encoder_2d(x_dict['2d'])
        z_3d = self.encoder_3d(x_dict['3d'])
        return F.normalize(z_2d, p=2, dim=-1), F.normalize(z_3d, p=2, dim=-1)