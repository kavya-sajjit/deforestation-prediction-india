#!/usr/bin/env python3
"""
CLI Script to Run Inference and Generate B-cos Explanations on Any Satellite Patch.

Usage:
    python run_inference.py --image_path data/satellite/india_sentinel2/test/deforested/odisha_forest_mines_p0001_deforested.png
    python run_inference.py --random_test
"""

import argparse
import os
import random
from PIL import Image
import torch
from torchvision import transforms

from src.explain.bcos_explainer import BcosExplainer
from src.explain.visualize import plot_single_explanation
from src.models.bcos_cnn import BcosDeforestationCNN


def main():
    parser = argparse.ArgumentParser(description="Run Deforestation Prediction & B-cos Interpretability")
    parser.add_argument("--image_path", type=str, default=None, help="Path to input satellite image")
    parser.add_argument("--model_path", type=str, default="models/bcos_deforestation_cnn.pth", help="Trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/satellite/india_sentinel2", help="Dataset directory (for random test)")
    parser.add_argument("--random_test", action="store_true", help="Pick a random sample from test dataset")
    parser.add_argument("--output_path", type=str, default="models/inference_explanation.png", help="Where to save explanation figure")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Load model
    print(f"Loading B-cos Deforestation CNN from {args.model_path}...")
    model = BcosDeforestationCNN(in_channels=3, num_classes=2, base_channels=32, b=2, auto_add_inverse=True)
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Warning: checkpoint not found at {args.model_path}, using randomly initialized model.")
    model.to(device)
    model.eval()

    # 2. Select image
    img_path = args.image_path
    if args.random_test or img_path is None:
        test_dir = os.path.join(args.data_dir, "test")
        candidates = []
        for root, _, files in os.walk(test_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    candidates.append(os.path.join(root, f))
        if not candidates:
            print(f"Error: No images found in {test_dir}. Provide --image_path directly.")
            return
        img_path = random.choice(candidates)

    print(f"Analyzing satellite patch: {img_path}")
    raw_img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    tensor = transform(raw_img)

    # 3. Explain with B-cos
    explainer = BcosExplainer(model, device=device)
    class_names = ["Intact Forest", "Deforested"]
    exp = explainer.explain(tensor, class_names=class_names)

    print("\n============== INFERENCE RESULTS ==============")
    print(f"Predicted Class:            {exp['target_class_name']}")
    print(f"Confidence:                 {exp['probability']:.2%}")
    print(f"Class Logit Value:          {exp['logit']:.4f}")
    print(f"Linear Reconstruction Sum:  {exp['linear_reconstruction_sum']:.4f}")
    print(f"Exact Mathematical Fidelity: Error = {exp['fidelity_error']:.2e}")
    print("===============================================")

    # 4. Save explanation figure
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    plot_single_explanation(exp, save_path=args.output_path)
    print(f"Explanation plot saved to: {args.output_path}")


if __name__ == "__main__":
    main()
