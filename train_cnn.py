#!/usr/bin/env python3
"""
CLI Script to Train and Evaluate B-cos and Baseline CNNs on Real Satellite Imagery.

Usage:
    python train_cnn.py --model bcos --epochs 15 --batch_size 32
    python train_cnn.py --model baseline --epochs 15
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.data.satellite_dataset import build_dataloaders
from src.data.stac_downloader import build_sentinel2_dataset
from src.explain.bcos_explainer import BcosExplainer
from src.explain.visualize import plot_explanation_gallery, plot_single_explanation
from src.models.baseline_cnn import StandardDeforestationCNN
from src.models.bcos_cnn import BcosDeforestationCNN
from src.training.train_bcos import evaluate_model, plot_training_curves, train_deforestation_model


def main():
    parser = argparse.ArgumentParser(description="Train Deforestation CNN with B-cos on Real Satellite Data")
    parser.add_argument("--model", type=str, choices=["bcos", "baseline"], default="bcos", help="Model type to train")
    parser.add_argument("--data_dir", type=str, default="data/satellite/india_sentinel2", help="Satellite dataset directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--b", type=int, default=2, help="B-cos scaling exponent (typically 2)")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models and figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Check if real satellite dataset exists; acquire if missing
    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Satellite dataset not found at {args.data_dir}. Downloading real Sentinel-2 scenes for India...")
        build_sentinel2_dataset(args.data_dir)

    # 2. Build DataLoaders
    print("\n--- Initializing DataLoaders ---")
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, img_size=128
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 3. Instantiate model
    if args.model == "bcos":
        print(f"\n--- Initializing Interpretable B-cos CNN (B={args.b}) ---")
        model = BcosDeforestationCNN(in_channels=3, num_classes=2, base_channels=32, b=args.b, auto_add_inverse=True)
        ckpt_name = "bcos_deforestation_cnn.pth"
    else:
        print("\n--- Initializing Standard Baseline CNN ---")
        model = StandardDeforestationCNN(in_channels=3, num_classes=2, base_channels=32)
        ckpt_name = "baseline_cnn.pth"

    ckpt_path = os.path.join(args.output_dir, ckpt_name)

    # 4. Train model
    print(f"\n--- Starting Training ({args.epochs} epochs) ---")
    history = train_deforestation_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        checkpoint_path=ckpt_path,
    )

    # 5. Plot and save learning curves
    curves_path = os.path.join(args.output_dir, f"{args.model}_training_curves.png")
    plot_training_curves(history, save_path=curves_path)
    print(f"Training curves saved to: {curves_path}")

    # 6. Load best checkpoint and evaluate on test set
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_metrics, cm, y_true, y_pred, y_probs = evaluate_model(
        model, test_loader, criterion, device
    )

    print("\n================ TEST SET EVALUATION ================")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Accuracy:       {test_metrics['accuracy']:.2%}")
    print(f"Precision:      {test_metrics['precision']:.3f}")
    print(f"Recall:         {test_metrics['recall']:.3f}")
    print(f"F1-Score:       {test_metrics['f1']:.3f}")
    print(f"ROC-AUC:        {test_metrics.get('roc_auc', 0.0):.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("=====================================================")

    # Plot Confusion Matrix
    cm_path = os.path.join(args.output_dir, f"{args.model}_confusion_matrix.png")
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Intact Forest", "Deforested"],
        yticklabels=["Intact Forest", "Deforested"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Ground Truth")
    plt.title(f"{args.model.upper()} Confusion Matrix on Real Sentinel-2 Test Patches")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # 7. For B-cos model, generate interpretability explanations on test patches
    if args.model == "bcos":
        print("\n--- Generating Faithful B-cos Explanations ---")
        explainer = BcosExplainer(model, device=device)
        class_names = ["Intact Forest", "Deforested"]
        gallery_exps = []

        # Sample representative test patches
        for idx in range(min(6, len(test_loader.dataset))):
            img_tensor, label, cls_name = test_loader.dataset[idx]
            exp = explainer.explain(img_tensor, class_names=class_names)
            gallery_exps.append(exp)
            print(
                f"Patch #{idx+1} ({cls_name}) -> Pred: {exp['target_class_name']} ({exp['probability']:.1%}) "
                f"| Fidelity error: {exp['fidelity_error']:.2e}"
            )

        gallery_path = os.path.join(args.output_dir, "bcos_explanation_gallery.png")
        plot_explanation_gallery(gallery_exps, save_path=gallery_path)
        print(f"B-cos explanation gallery saved to: {gallery_path}")

    print("\n--- All operations completed successfully! ---")


if __name__ == "__main__":
    main()
