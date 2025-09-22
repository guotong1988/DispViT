import torch
import torch.nn.functional as F
import einops

from dataclasses import dataclass


@dataclass(eq=False)
class DispLoss(torch.nn.Module):
    def __init__(self, disp, logits):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.disp = disp
        self.logits = logits

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks

        Returns:
            Dict containing individual losses and total objective
        """
        gt_disp = batch["disp"]
        valid = torch.logical_and(batch["valid"], gt_disp < self.disp["max_disp"])

        loss_disp = disp_loss(predictions["disp"], gt_disp, valid)
        loss_logits = disp_softmax(predictions["disp_logits"], gt_disp, valid)

        loss_dict = {
            "objective": self.disp["weight"]*loss_disp + self.logits["weight"]*loss_logits,
            "loss_disp": loss_disp,
            "loss_logits": loss_logits,
        }
        return loss_dict


def disp_loss(pred_disp, target_disp, mask):
    pred_disp = pred_disp.float()
    target_disp = target_disp.float()
    error = torch.abs(pred_disp - target_disp)
    loss = (error * mask).sum() / (mask.sum() + 1e-6)
    return loss


def disp_softmax(logits, labels, mask):
    logits = logits.float()
    labels = labels.float()
    labels = labels.clamp(0, 381.0)
    interval = 381.0 / 127.0
    labels = labels.unsqueeze(1) / interval
    lb = torch.floor(labels).long()
    hb = (1 + lb).clamp_max(127)
    labels_ = torch.zeros_like(logits)
    wh = labels - lb
    labels_.scatter_add_(1, lb, 1.0 - wh)
    labels_.scatter_add_(1, hb, wh)
    loss = F.cross_entropy(logits, labels_, reduction='none')
    mask = mask.float()
    loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    return loss


@dataclass(eq=False)
class RefineLoss(torch.nn.Module):
    def __init__(self, disp, loss_type, **kwargs_ignored):
        super().__init__()
        self.disp = disp
        self.weights = disp["weight"]
        if loss_type not in ['l1', 'smooth_l1']:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.criterion = F.l1_loss
        else:
            self.criterion = F.smooth_l1_loss

    def _loss_fn(self, pred, target, mask):
        if torch.any(mask):
            loss = self.criterion(pred[mask], target[mask], reduction='mean')
        else:
            loss = F.smooth_l1_loss(pred, pred.detach(), reduction='mean')
        return loss
    
    def _loss_aux(self, fmap1, fmap2, target):
        fmap1 = F.avg_pool2d(fmap1, kernel_size=2, stride=2)
        fmap2 = F.avg_pool2d(fmap2, kernel_size=2, stride=2)
        cost_volume = build_correlation_volume(fmap1, fmap2, 40)
        prob = cost_volume.softmax(dim=1).permute(0, 2, 3, 1).flatten(0, 2)  # [BHW,40]

        B, H, W = target.shape
        target = torch.clamp(target, min=0)
        valid = (target > 0) & (target < 320)

        ref = torch.arange(0, W, dtype=torch.long, device=prob.device).reshape(1, 1, -1).expand(B, H, W)
        coord = ref - target  # corresponding coordinate in the right view
        valid = torch.logical_and(valid, coord >= 0)  # correspondence should within image boundary

        # scale ground truth disparity
        tgt = target / 8

        weights = torch.ones_like(tgt)
        weights[~valid] = 0.0

        N = 40
        tgt = einops.rearrange(tgt, 'b (h m) (w n) -> (b h w) (m n)', m=8, n=8)
        weights = einops.rearrange(weights, 'b (h m) (w n) -> (b h w) (m n)', m=8, n=8)
        valid = einops.rearrange(valid, 'b (h m) (w n) -> (b h w) (m n)', m=8, n=8)
        lower_bound = torch.floor(tgt).to(torch.long)
        high_bound = lower_bound + 1
        high_prob = tgt - lower_bound
        lower_bound = torch.clamp(lower_bound, max=N - 1)
        high_bound = torch.clamp(high_bound, max=N - 1)

        lower_prob = (1 - high_prob) * weights
        high_prob = high_prob * weights

        label = torch.zeros_like(prob)
        label.scatter_reduce_(dim=-1, index=lower_bound, src=lower_prob, reduce="sum")
        label.scatter_reduce_(dim=-1, index=high_bound, src=high_prob, reduce="sum")

        # normalize weights
        normalizer = torch.clamp(torch.sum(label, dim=-1, keepdim=True), min=1e-3)
        label = label / normalizer

        mask = label > 0
        log_prob = -(torch.log(torch.clamp(prob[mask], min=1e-5)) * label[mask]).sum()
        valid_pixs = (valid.float().sum(dim=-1) > 0).sum()
        return log_prob / (valid_pixs + 1e-5)

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks

        Returns:
            Dict containing individual losses and total objective
        """
        gt_disp = batch["disp"]
        valid = torch.logical_and(batch["valid"], gt_disp < self.disp["max_disp"])

        if "fmap1" in predictions and "fmap2" in predictions:
            loss_aux = self._loss_aux(predictions["fmap1"], predictions["fmap2"], gt_disp)
        else:
            loss_aux = 0.0

        disp_preds = predictions["disp_all"]
        loss_disp = [self._loss_fn(pred, gt_disp, valid) for pred in disp_preds]
        assert len(loss_disp) == len(self.weights), f"Expected {len(self.weights)} disparity predictions, but got {len(loss_disp)}"
        loss_dict = {
            "objective": sum(w * l for w, l in zip(self.weights, loss_disp)) + self.disp["aux_weight"] * loss_aux,
            "loss_disp": loss_disp[-1],
            "loss_aux": loss_aux,
            **{f"loss_disp_{i}": l for i, l in enumerate(loss_disp)},
        }
        return loss_dict
    

def build_correlation_volume(refimg_fea, targetimg_fea, max_disp):
    """ Build correlation volume for cost volume construction
    Args:
        refimg_fea: the feature map of reference image, [B,C,H,W]
        targetimg_fea: the feature map of target image, [B,C,H,W]
        max_disp: the maximum disparity
    Returns:
        cost_volume: the correlation volume, [B, max_disp, H, W]
    """
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros(B, max_disp, H, W)
    for i in range(max_disp):
        if i > 0:
            volume[:, i, :, i:] = (refimg_fea[:, :, :, i:] * targetimg_fea[:, :, :, :-i]).mean(dim=1)
        else:
            volume[:, i, :, :] = (refimg_fea * targetimg_fea).mean(dim=1)
    return volume