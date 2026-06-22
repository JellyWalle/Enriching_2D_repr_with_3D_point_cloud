import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

class HybridGNN(nn.Module):
    def __init__(self, input_dim_2d: int = 1028, input_dim_3d: int = 18, hidden_dim: int = 128, num_layers: int = 2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # 1. Сбалансированные энкодеры с LayerNorm (исправляет дисбаланс 1028 vs 18)
        self.encoder_2d = nn.Sequential(
            nn.Linear(input_dim_2d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.encoder_3d = nn.Sequential(
            nn.Linear(input_dim_3d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # 2. ВАЖНО: Только топологические рёбра. correspondence УДАЛЕНЫ отсюда (нет утечки данных)
            conv = HeteroConv({
                ('2d', 'adjacent_2d', '2d'): SAGEConv(hidden_dim, hidden_dim),
                ('3d', 'adjacent_3d_convex', '3d'): SAGEConv(hidden_dim, hidden_dim),
                ('3d', 'adjacent_3d_concave', '3d'): SAGEConv(hidden_dim, hidden_dim),
                ('3d', 'adjacent_3d_orthogonal', '3d'): SAGEConv(hidden_dim, hidden_dim),
                ('3d', 'adjacent_3d_unknown', '3d'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='mean')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        h_dict = {
            '2d': self.encoder_2d(x_dict['2d']),
            '3d': self.encoder_3d(x_dict['3d'])
        }

        for conv in self.convs:
            out_dict = conv(h_dict, edge_index_dict)
            for key in h_dict:
                # Residual connection: если ключа нет, берем ноль
                out = out_dict.get(key, torch.zeros_like(h_dict[key]))
                h_dict[key] = h_dict[key] + out
                
                # Нелинейность и L2-нормализация (критически важно для NT-Xent)
                h_dict[key] = F.relu(h_dict[key])
                h_dict[key] = self.dropout(h_dict[key]) 
                h_dict[key] = F.normalize(h_dict[key], p=2, dim=-1)

        return h_dict['2d'], h_dict['3d']


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_2d: torch.Tensor, z_3d: torch.Tensor, corr_edge_index: torch.Tensor):
        if corr_edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=z_2d.device)

        sorted_indices = torch.argsort(corr_edge_index[0])
        idx_2d = corr_edge_index[0][sorted_indices]
        idx_3d = corr_edge_index[1][sorted_indices]

        z_pos_2d = F.normalize(z_2d[idx_2d], p=2, dim=1)
        z_pos_3d = F.normalize(z_3d[idx_3d], p=2, dim=1)

        logits = torch.matmul(z_pos_2d, z_pos_3d.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)

        return F.cross_entropy(logits, labels)