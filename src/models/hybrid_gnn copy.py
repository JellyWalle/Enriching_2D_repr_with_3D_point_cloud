import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv

class HybridGNN(nn.Module):
    """
    Этап 5.1: Архитектура модели (Аналогия: PocketFinderGNN + GC-CAD)
    """
    def __init__(self, input_dim_2d: int = 1028, input_dim_3d: int = 18, hidden_dim: int = 128, num_layers: int = 2):
        super(HybridGNN, self).__init__()
        self.hidden_dim = hidden_dim

        # 1. Энкодеры для приведения к общему скрытому пространству
        self.encoder_2d = nn.Sequential(
            nn.Linear(input_dim_2d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.encoder_3d = nn.Sequential(
            nn.Linear(input_dim_3d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. Слои графовой свёртки (HeteroConv)
        # Мы динамически создаем словари свёрток для каждого типа связи
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # 2D-2D связи
                ('2d', 'adjacent_2d', '2d'): SAGEConv(hidden_dim, hidden_dim),
                
                # 3D-3D связи (Разные веса для convex, concave, orthogonal - это ключевая фишка!)
                ('3d', 'adjacent_3d_convex', '3d'): SAGEConv(hidden_dim, hidden_dim),
                ('3d', 'adjacent_3d_concave', '3d'): SAGEConv(hidden_dim, hidden_dim),
                ('3d', 'adjacent_3d_orthogonal', '3d'): SAGEConv(hidden_dim, hidden_dim),
                ('3d', 'adjacent_3d_unknown', '3d'): SAGEConv(hidden_dim, hidden_dim),
                
                # 2D-3D связи (correspondence) - позволяют обмениваться информацией между модальностями
                ('2d', 'correspondence', '3d'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('3d', 'rev_correspondence', '2d'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        # 1. Начальное кодирование признаков
        h_2d = self.encoder_2d(x_dict['2d'])
        h_3d = self.encoder_3d(x_dict['3d'])
        h_dict = {'2d': h_2d, '3d': h_3d}

        # Добавляем обратные рёбра correspondence для двустороннего обмена сообщениями
        if ('2d', 'correspondence', '3d') in edge_index_dict:
            edge_index_dict['3d', 'rev_correspondence', '2d'] = edge_index_dict['2d', 'correspondence', '3d'].flip([0])

        # 2. Графовые свёртки
        for conv in self.convs:
            # HeteroConv автоматически применяет нужную свёртку к нужному типу ребра
            h_dict = conv(h_dict, edge_index_dict)
            # Применяем нелинейность и нормализацию для стабильности
            for key in h_dict:
                h_dict[key] = F.relu(h_dict[key])
                h_dict[key] = F.normalize(h_dict[key], p=2, dim=-1) # Важно для контрастной потери

        return h_dict['2d'], h_dict['3d']


class NTXentLoss(nn.Module):
    """
    Этап 6.2: Функция потерь NT-Xent (InfoNCE) для кросс-модального сопоставления.
    Сближает эмбеддинги пар, связанных ребром 'correspondence', и отдаляет все остальные.
    """
    def __init__(self, temperature: float = 0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_2d: torch.Tensor, z_3d: torch.Tensor, corr_edge_index: torch.Tensor):
        """
        z_2d: (N_2d, hidden_dim)
        z_3d: (N_3d, hidden_dim)
        corr_edge_index: (2, E_corr) - индексы узлов, которые должны соответствовать друг другу
        """
        if corr_edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=z_2d.device)

        # 1. Извлекаем эмбеддинги только тех узлов, которые участвуют в correspondence
        # Сортируем по 2d-узлам, чтобы положительные пары шли по диагонали
        sorted_indices = torch.argsort(corr_edge_index[0])
        idx_2d = corr_edge_index[0][sorted_indices]
        idx_3d = corr_edge_index[1][sorted_indices]

        z_pos_2d = z_2d[idx_2d]  # (E, hidden_dim)
        z_pos_3d = z_3d[idx_3d]  # (E, hidden_dim)

        # Нормализация (уже сделана в forward, но на всякий случай)
        z_pos_2d = F.normalize(z_pos_2d, p=2, dim=1)
        z_pos_3d = F.normalize(z_pos_3d, p=2, dim=1)

        # 2. Вычисляем матрицу сходства (E x E)
        # logits[i, j] = similarity between 2d_node_i and 3d_node_j
        logits = torch.matmul(z_pos_2d, z_pos_3d.T) / self.temperature

        # 3. Целевые метки: положительные пары находятся на главной диагонали
        labels = torch.arange(logits.shape[0], device=logits.device)

        # 4. Вычисляем Cross-Entropy Loss
        # F.cross_entropy автоматически применяет log_softmax и NLLLoss
        loss = F.cross_entropy(logits, labels)
        
        return loss