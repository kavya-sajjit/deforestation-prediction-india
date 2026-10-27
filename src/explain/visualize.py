"""
Visualization Utilities for Satellite Deforestation Predictions & B-cos Explanations.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_single_explanation(
    exp_dict: dict,
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
):
    """
    Plots a 3-panel explanation for a satellite image patch:
    Panel 1: Original Sentinel-2 True Color
    Panel 2: B-cos Linear Contribution Heatmap (Spatial Impact)
    Panel 3: B-cos Decoded Visual Explanation (Feature Evidence)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Original Satellite Patch
    rgb = np.clip(exp_dict["input_rgb"], 0.0, 1.0)
    axes[0].imshow(rgb)
    axes[0].set_title(
        f"Input Satellite Patch\nPred: {exp_dict['target_class_name']} ({exp_dict['probability']:.1%})",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].axis("off")

    # 2. Spatial Contribution Heatmap
    contrib = exp_dict["spatial_contributions"]
    vmax = np.percentile(np.abs(contrib), 99)
    vmax = max(vmax, 1e-4)
    im1 = axes[1].imshow(contrib, cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[1].set_title(
        f"Spatial Contribution Map\nFidelity Error: {exp_dict['fidelity_error']:.1e}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Decoded B-cos Feature Map
    rgba = exp_dict["rgba_explanation"]
    axes[2].imshow(rgba)
    axes[2].set_title("B-cos Decoded Evidence\n(Positive Alignments)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return fig


def plot_explanation_gallery(
    explanations: List[dict],
    save_path: Optional[str] = None,
    title: str = "B-cos Interpretable Deforestation Predictions on Real Sentinel-2 Satellite Data",
):
    """
    Plots a multi-row gallery comparing multiple satellite patches.
    """
    n = len(explanations)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    for i, exp in enumerate(explanations):
        # Panel 1: Satellite Image
        rgb = np.clip(exp["input_rgb"], 0.0, 1.0)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(
            f"Patch #{i+1}: {exp['target_class_name']} ({exp['probability']:.1%})",
            fontsize=10,
        )
        axes[i, 0].axis("off")

        # Panel 2: Contribution Heatmap
        contrib = exp["spatial_contributions"]
        vmax = max(np.percentile(np.abs(contrib), 99), 1e-4)
        im = axes[i, 1].imshow(contrib, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[i, 1].set_title(
            f"Linear Contribution (Δ = {exp['fidelity_error']:.2e})",
            fontsize=10,
        )
        axes[i, 1].axis("off")
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Panel 3: Decoded Visual Evidence
        axes[i, 2].imshow(exp["rgba_explanation"])
        axes[i, 2].set_title("B-cos Explanatory Attribution", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return fig
