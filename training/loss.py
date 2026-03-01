import torch
import torch.nn.functional as F
import einops

from dataclasses import dataclass

from dispvit.depth_anything.depth_anything import DepthAnything


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


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
    error = torch.abs(pred_disp[mask] - target_disp[mask])
    loss = error.sum() / (mask.sum() + 1e-6)
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
    def __init__(self, disp, loss_type, regress, gram, **kwargs_ignored):
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
        self.regress = regress
        self.gram = gram

        # pretrained_model_name_or_path = 'depth_anything_v2_vitl.pth'
        # self.dav2 = DepthAnything.from_pretrained(pretrained_model_name_or_path)
        # self.dav2.freeze()

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

    def _loss_fn(self, pred, target, mask):
        if torch.any(mask):
            loss = self.criterion(pred[mask], target[mask], reduction='mean')
        else:
            loss = F.smooth_l1_loss(pred, pred.detach(), reduction='mean')
        return loss
    
    def gram_loss(self, pred, batch):
        img1 = (batch["img1"] / 255.0 - self._resnet_mean) / self._resnet_std

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                target_feats = self.dav2.pretrained.get_intermediate_layers(img1, self.dav2.intermediate_layer_idx[self.dav2.encoder], return_class_token=True)
        student_feats = pred['gram_feats']

        loss = []
        for target_feat, student_feat in zip(target_feats, student_feats):
            target_feat = target_feat[0]
            student_feat = student_feat[0]
            loss.append(gram_loss_fn(student_feat, target_feat))
        return sum(loss) / len(loss)

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

        disp_preds = predictions["disp_all"]
        loss_disp = [self._loss_fn(pred, gt_disp, valid) for pred in disp_preds]
        # disp_vit_preds = predictions["disp_vit"]
        # loss_disp_vit = [self._loss_fn(pred, gt_disp, valid) for pred in disp_vit_preds]
        # assert len(disp_preds) == len(disp_vit_preds), f"Expected the same number of predictions from both branches, but got {len(disp_preds)} and {len(disp_vit_preds)}"
        # loss_disp = [l1 + l2 for l1, l2 in zip(loss_disp_, loss_disp_vit)]
        assert len(loss_disp) == len(self.weights), f"Expected {len(self.weights)} disparity predictions, but got {len(loss_disp)}"

        loss_disp_regress = disp_loss(predictions["disp_regress"], gt_disp, valid)
        loss_logits = disp_softmax(predictions["disp_logits"], gt_disp, valid)

        # Gram loss computation
        #loss_gram_anchoring = self.gram_loss(predictions, batch)

        loss_dict = {
            "objective": sum(w * l for w, l in zip(self.weights, loss_disp)) + loss_disp_regress * self.regress["disp_weight"] + loss_logits * self.regress["logit_weight"],
            "loss_disp": loss_disp[-1],
            # "loss_disp_vit": loss_disp_vit[-1],
            "loss_disp_regress": loss_disp_regress,
            **{f"loss_disp_{i}": l for i, l in enumerate(loss_disp)},
        }
        return loss_dict
    

def gram_loss_fn(output_feats, target_feats):
    """Compute the MSE loss between the gram matrix of the input and target features.
    
    Args:
        output_feats: Pytorch tensor (B,N,dim)
        target_feats: Pytorch tensor (B,N,dim)
    """
    # Float casting
    output_feats = output_feats.float()
    target_feats = target_feats.float()

    target_feats = F.normalize(target_feats, dim=-1)
    # Compute similarities
    target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

    output_feats = F.normalize(output_feats, dim=-1)
    # Compute similarities
    output_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

    loss = F.mse_loss(output_sim, target_sim)
    return loss