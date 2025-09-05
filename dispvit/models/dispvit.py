import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_head import DPTHead

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class DispViT(nn.Module):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    def __init__(self, encoder_type="vitl", init_weights=True):
        super().__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }

        self.encoder_type = encoder_type

        dinov2_model = getattr(importlib.import_module("..layers.dinov2", __package__), f"dinov2_{encoder_type}14")
        encoder = dinov2_model(pretrained=init_weights)

        # Reuse the pretrained Conv2d weights of patch embed layer and make it work with 6 input channels
        # by duplicating the weights tensor of the proj layer and divide its value by two.
        self.__build_patch_embed__(encoder.patch_embed)
        self.encoder = encoder

        self.dpt_head = DPTHead(self.encoder.embed_dim, patch_size=self.encoder.patch_size, output_dim=256,
                                features=self.model_configs[encoder_type]["features"], 
                                hidden_dims=[128, 128],
                                out_channels=self.model_configs[encoder_type]["out_channels"])
        
        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)
        
    def __build_patch_embed__(self, patch_embed):
        new_proj = nn.Conv2d(
            6,
            patch_embed.proj.out_channels,
            kernel_size=patch_embed.proj.kernel_size,
            stride=patch_embed.proj.stride,
        )
        with torch.no_grad():
            new_proj.weight[:, :3, :, :] = patch_embed.proj.weight / 2
            new_proj.weight[:, 3:, :, :] = patch_embed.proj.weight / 2
            if patch_embed.proj.bias is not None:
                new_proj.bias.copy_(patch_embed.proj.bias)
        patch_embed.proj = new_proj

    def forward(self, batch):
        img1 = (batch["img1"] / 255.0 - self._resnet_mean) / self._resnet_std
        img2 = (batch["img2"] / 255.0 - self._resnet_mean) / self._resnet_std
        img = torch.cat((img1, img2), dim=1)

        padder = None
        if not self.training:
            padder = InputPadder(img.shape, mode="nmrf", divis_by=self.encoder.patch_size)
            img = padder.pad(img)[0]

        B, C_in, H, W = img.shape

        if C_in != 6:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        
        # Normalize images
        patch_h, patch_w = H // self.encoder.patch_size, W // self.encoder.patch_size

        features = self.encoder.get_intermediate_layers(img, self.intermediate_layer_idx[self.encoder_type], return_class_token=True)

        disp, disp_logits = self.prediction_head(features, patch_h, patch_w)
        if padder is not None:
            disp = padder.unpad(disp.unsqueeze(1)).squeeze(1)
            disp_logits = padder.unpad(disp_logits)
        return {"disp": disp, "disp_logits": disp_logits}

    def prediction_head(self, x, patch_h, patch_w):
        soft_argmax_threshold = 7
        softmax_temperature = 0.5
        disp_logits = self.dpt_head(x, patch_h, patch_w)
        argmax_w = disp_logits.argmax(
            dim=1, keepdim=True
        )
        index = torch.arange(disp_logits.shape[1], device=disp_logits.device).view(1, -1, 1, 1)
        mask = (torch.abs(argmax_w - index) <= soft_argmax_threshold).float()
        probs = F.softmax(disp_logits * softmax_temperature, dim=1) * mask
        probs = probs / probs.sum(dim=1, keepdim=True)

        ticks = torch.linspace(0, 765, 256, dtype=torch.float32, device=disp_logits.device).view(1, -1, 1, 1)
        disp = torch.sum(probs * ticks, dim=1)
        return disp, disp_logits


class InputPadder:
    """ Pads images such that dimensions are divisible by given factor """

    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'nmrf':
            self._pad = [0, pad_wd, pad_ht, 0]
        else:
            raise ValueError(f"Non recognized mode '{mode}'")

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]