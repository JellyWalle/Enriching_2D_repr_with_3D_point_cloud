import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from torchvision.ops import roi_align

class PointNet(nn.Module):
    def __init__(self, input_dim=3, out_dim=256):
        super().__init__()
        self.mlp1 = nn.Sequential(
            # [B, 3, N] -> [B, 64, N]
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # [B, 64, N] -> [B, 128, N]
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # [B, 128, N] -> [B, 256, N]
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        # x: [B, N, 3] → [B, 3, N]
        if x.dim() == 3:
            x = x.transpose(1, 2).contiguous()
        features = self.mlp1(x)  # [B, 256, N]
        out = torch.max(features, dim=2).values  # [B, 256]
        return out

# -------------------------------------------------
# Основная модель
# -------------------------------------------------
class YOLOv8_3DFusion(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        self.pointnet = PointNet(input_dim=3, out_dim=256)
        self.orientation_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.roi_out_size = 7
        self.fusion_proj = nn.Linear(64 * self.roi_out_size * self.roi_out_size + 256, 64)

    def forward(self, x, bbox_abs=None, pointclouds=None, img_hw=None):
        print("start forward")

        print(bbox_abs)
        print(pointclouds)
        # Обычный inference (без 3D)
        if bbox_abs is None or pointclouds is None:
            print('only detection')
            return super().forward(x)

        # === Backbone ===
        x = self.model[0](x)
        x = self.model[1](x)
        x = self.model[2](x)
        c2_32 = x  # P2
        x = self.model[3](x)
        x = self.model[4](x)
        c3_64 = x  # P3
        x = self.model[5](x)
        x = self.model[6](x)
        c4_128 = x  # P4
        x = self.model[7](x)
        x = self.model[8](x)
        x = self.model[9](x)
        c5_256 = x  # P5

        # === Neck (FPN + PAN) ===
        up5 = self.model[10](c5_256)
        cat1 = self.model[11]([up5, c4_128])
        c4_fpn = self.model[12](cat1)

        up4 = self.model[13](c4_fpn)
        cat2 = self.model[14]([up4, c3_64])
        c3_out = self.model[15](cat2)

        down3 = self.model[16](c3_out)
        cat3 = self.model[17]([down3, c4_fpn])
        c4_out = self.model[18](cat3)

        down4 = self.model[19](c4_out)
        cat4 = self.model[20]([down4, c5_256])
        c5_out = self.model[21](cat4)

        feats = [c3_out, c4_out, c5_out]
        p3 = feats[0]  # [B, 64, H, W]
        B, C_p3, H_f, W_f = p3.shape
        assert C_p3 == 64, f"Expected 64 channels in p3, got {C_p3}"

        # ---------- Этап 2: Масштабирование bbox ----------
        orig_H = img_hw[:, 0].float()  # [B]
        orig_W = img_hw[:, 1].float()  # [B]
        scale_h = x.shape[2] / orig_H
        scale_w = x.shape[3] / orig_W

        boxes_scaled = bbox_abs.clone().float()  # [B, 4]
        boxes_scaled[:, [0, 2]] *= scale_w.unsqueeze(1)
        boxes_scaled[:, [1, 3]] *= scale_h.unsqueeze(1)

        # ---------- Этап 3: ROI Align ----------
        rois_list = []
        for i in range(B):
            box = boxes_scaled[i].unsqueeze(0)  # [1, 4]
            roi = roi_align(p3[i:i+1], [box], output_size=self.roi_out_size, aligned=True)
            rois_list.append(roi.flatten())
        f_2d = torch.stack(rois_list)  # [B, 64 * 7 * 7]

        # ---------- Этап 4: 3D признаки ----------
        f_3d_list = []
        for i in range(B):
            pc = pointclouds[i]  # [N, 3]
            if pc.numel() == 0:
                f3d = torch.zeros(256, device=pc.device)
            else:
                f3d = self.pointnet(pc.unsqueeze(0)).squeeze(0)  # [256]
            f_3d_list.append(f3d)
        f_3d = torch.stack(f_3d_list)  # [B, 256]

        # ---------- Этап 5: Fusion ----------
        fused = self.fusion_proj(torch.cat([f_2d, f_3d], dim=1))  # [B, 64]
        fused = F.dropout(fused, p=0.1, training=self.training)

        # ---------- Этап 6: Встраивание в p3 ----------
        fused_expanded = fused.view(B, 64, 1, 1).expand(B, 64, H_f, W_f)
        p3_enhanced = p3 + fused_expanded
        feats[0] = p3_enhanced

        # ---------- Этап 7: Выход ----------
        det_out = self.model[22](feats)  # Detect head
        quat_pred = self.orientation_head(fused)  # [B, 4]
        quat_pred = F.normalize(quat_pred, p=2, dim=1)
        print("in model quat_pred shape = : ", quat_pred.size())

        return det_out, quat_pred