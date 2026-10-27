"""
B-cos Model Explanation and Interpretability Engine.
Extracts mathematically faithful linear contribution maps from B-cos networks.
Reference: Böhle et al., CVPR 2022 (https://arxiv.org/abs/2205.10268)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from ..models.bcos_modules import AddInverse, set_explanation_mode


def decode_bcos_rgb(
    x_inv: torch.Tensor,
    linear_mapping: torch.Tensor,
    smooth: int = 5,
    alpha_percentile: float = 99.0,
) -> np.ndarray:
    """
    Decodes the 6-channel linear mapping W_{1->L} into an RGB explanation image
    with an alpha channel indicating the spatial magnitude of positive contribution.
    
    Args:
        x_inv: 6-channel input tensor [6, H, W] (RGB + (1 - RGB))
        linear_mapping: Gradient/weight tensor [6, H, W]
        smooth: Kernel size for alpha smoothing
        alpha_percentile: Percentile threshold for alpha clipping
        
    Returns:
        RGBA numpy array [H, W, 4] with values in [0, 1]
    """
    # [H, W] spatial contribution
    contribs = (x_inv * linear_mapping).sum(0)

    # Normalize each channel pair maintaining direction
    max_val = linear_mapping.abs().max(0, keepdim=True)[0] + 1e-12
    rgb_grad = linear_mapping / max_val
    rgb_grad = rgb_grad.clamp(min=0)

    # Normalize RGB vs (1 - RGB)
    denom = rgb_grad[:3] + rgb_grad[3:] + 1e-12
    rgb_colors = (rgb_grad[:3] / denom).detach().cpu().numpy()

    # Alpha: strength of positive contributions
    alpha = linear_mapping.norm(p=2, dim=0, keepdim=True)
    alpha = torch.where(contribs[None] <= 0, torch.zeros_like(alpha) + 1e-12, alpha)

    if smooth and smooth > 1:
        padding = (smooth - 1) // 2
        alpha = F.avg_pool2d(alpha.unsqueeze(0), smooth, stride=1, padding=padding).squeeze(0)

    alpha_np = alpha[0].detach().cpu().numpy()
    cutoff = np.percentile(alpha_np, alpha_percentile)
    if cutoff > 1e-8:
        alpha_np = np.clip(alpha_np / cutoff, 0.0, 1.0)
    else:
        alpha_np = np.zeros_like(alpha_np)

    rgba = np.concatenate([rgb_colors, alpha_np[None, ...]], axis=0)
    return rgba.transpose((1, 2, 0))


class BcosExplainer:
    """
    Explainer for B-cos models that computes the exact linear decomposition
    of the model output without perturbation, sampling, or gradient approximations.
    """
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.add_inverse = AddInverse(dim=1)

    def explain(
        self,
        img_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        class_names: Optional[list] = None,
    ) -> Dict:
        """
        Explain a single satellite image prediction.
        
        Args:
            img_tensor: Single image tensor [3, H, W] or batch [1, 3, H, W] with values in [0, 1]
            target_class: Class index to explain (defaults to argmax predicted class)
            class_names: Optional list of class strings
            
        Returns:
            Dictionary containing prediction, probabilities, fidelity check,
            spatial contribution map, and decoded RGBA explanation.
        """
        self.model.eval()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # Ensure AddInverse 6-channel input with requires_grad
        x_inv = self.add_inverse(img_tensor).detach().requires_grad_(True)

        # 1. Standard forward pass for logits
        set_explanation_mode(self.model, False)
        with torch.no_grad():
            std_logits = self.model(img_tensor)
            probs = torch.softmax(std_logits, dim=-1)[0]
            pred_class = int(std_logits.argmax(dim=-1)[0].item())

        target = pred_class if target_class is None else target_class

        # 2. Explanation mode forward pass (detaching non-linear cosine scale)
        set_explanation_mode(self.model, True)
        # Call model on x_inv directly (bypassing auto_add_inverse)
        # If model expects 6 channels natively:
        exp_logits = self.model(x_inv)
        target_logit = exp_logits[0, target]

        # 3. Exact linear gradient computation W_{1->L}
        if x_inv.grad is not None:
            x_inv.grad.zero_()
        target_logit.backward()
        w = x_inv.grad[0]  # [6, H, W]

        # 4. Verify mathematical fidelity: logit == sum(x * w)
        linear_sum = float((x_inv[0] * w).sum().item())
        logit_val = float(target_logit.item())
        fidelity_error = abs(logit_val - linear_sum)

        # 5. Compute spatial contribution map C(h, w) = sum_c (x * w)
        spatial_contrib = (x_inv[0] * w).sum(dim=0).detach().cpu().numpy()

        # 6. Compute RGB decoded explanation
        rgba_explanation = decode_bcos_rgb(x_inv[0], w)

        # Reset explanation mode
        set_explanation_mode(self.model, False)

        c_name = class_names[target] if class_names and target < len(class_names) else f"Class {target}"

        return {
            "predicted_class": pred_class,
            "target_class": target,
            "target_class_name": c_name,
            "probability": float(probs[target].item()),
            "probabilities": probs.detach().cpu().numpy().tolist(),
            "logit": logit_val,
            "linear_reconstruction_sum": linear_sum,
            "fidelity_error": fidelity_error,
            "spatial_contributions": spatial_contrib,
            "rgba_explanation": rgba_explanation,
            "weights": w.detach().cpu().numpy(),
            "input_rgb": img_tensor[0].detach().cpu().permute(1, 2, 0).numpy(),
        }
