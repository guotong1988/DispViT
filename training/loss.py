import torch
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass(eq=False)
class DispLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks

        Returns:
            Dict containing individual losses and total objective
        """
        gt_disp = batch["disp"]
        B, H, W = gt_disp.shape
        coords_h, coords_w = torch.meshgrid(torch.arange(H, device=gt_disp.device), torch.arange(W, device=gt_disp.device), indexing='ij')
        target_coord = coords_w[None] - gt_disp
        valid = torch.logical_and(batch["valid"], gt_disp < 192)

        coord_loss = coordinate_loss(predictions["coord"], target_coord, valid)
        logits_loss = coordinate_softmax(predictions["coord_logits"], target_coord, valid)

        loss_dict = {
            "objective": 0.1*coord_loss + logits_loss,
            "loss_coord": coord_loss,
            "loss_logits": logits_loss,
        }
        return loss_dict


def coordinate_loss(pred_coord, target_coord, mask):
    pred_coord = pred_coord.float()
    target_coord = target_coord.float()
    error = torch.abs(pred_coord - target_coord)
    loss = (error * mask).sum() / (mask.sum() + 1e-6)
    return loss


def coordinate_softmax(logits, labels, mask):
    logits = logits.float()
    labels = labels.float()
    W = logits.shape[-1]
    labels += 0.1*W
    labels = labels.clamp(0, 1.1*W)
    interval = 1.1*W / 255.0
    labels = labels.unsqueeze(1) / interval
    lb = torch.floor(labels).long()
    hb = (1 + lb).clamp_max(255)
    labels_ = torch.zeros_like(logits)
    wh = labels - lb
    labels_.scatter_add_(1, lb, 1.0 - wh)
    labels_.scatter_add_(1, hb, wh)
    loss = F.cross_entropy(logits, labels_, reduction='none')
    mask = mask.float()
    loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    return loss