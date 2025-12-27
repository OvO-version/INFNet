from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from numpy import *
import warnings
import copy
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from utils.dataset import DatasetInit
from utils.utils import (load_checkpoint, save_checkpoint, basic_transform, train_transform, get_loaders, train_fn,
                         val_fn, compute_validation_loss, create_result_dir, save_predictions_as_imgs, get_result)

warnings.filterwarnings("ignore")

ENABLE_GRID_SEARCH = True
PARAM_GRID = {
    "learning_rate": [1e-5, 1e-4, 5e-4],
    "batch_size": [2, 4, 8],
    "loss_weight": [0.3, 0.5, 0.7]
}

NUM_FOLD = 5
NUM_EPOCHS = 100
IMG_HEIGHT = 224
IMG_WIDTH = 224
DATASET = "BUSI"
IMG_DIR = r"E:\DLPrograms\Project_ZYTWYX\Dataset_BUSI_with_GT\image"
MASK_DIR = r"E:\DLPrograms\Project_ZYTWYX\Dataset_BUSI_with_GT\mask"
SAVE_MODEL = False
USE_AMP = True
DEVICE = "cuda:0"
MODEL_NAME = "INFNet"

NUM_WORKERS = 0
PIN_MEMORY = True
DROP_LAST = True

from models.INFNet import INFNet

def get_loss_fn(weight=0.5):
    return nn.BCEWithLogitsLoss()

def compute_foreground_ratio(mask_dir):
    ratios = []
    for name in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        fg_pixels = np.sum(mask > 0)
        ratios.append(fg_pixels / mask.size)
    return np.array(ratios)

def plot_foreground_boxplot(mask_dir, save_dir, dataset_name):
    ratios = compute_foreground_ratio(mask_dir)
    plt.figure(figsize=(5, 4))
    plt.boxplot([ratios], labels=[dataset_name], showfliers=True)
    plt.ylabel("Foreground Pixel Ratio")
    plt.title("Foreground Pixel Ratio Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "Fig1_Foreground_Ratio_Boxplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Data Quality] Saved foreground ratio boxplot to {save_path}")

def visualize_image_mask_pairs(image_dir, mask_dir, save_dir, num_samples=3):
    names = random.sample(os.listdir(image_dir), min(num_samples, len(os.listdir(image_dir))))
    for idx, name in enumerate(names):
        img = cv2.imread(os.path.join(image_dir, name))
        mask = cv2.imread(os.path.join(mask_dir, name), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1); plt.imshow(img); plt.title("Image"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(mask, cmap="gray"); plt.title("Mask"); plt.axis("off")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"Quality_ImageMask_{idx}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_foreground_histogram(mask_dir, save_dir, dataset_name):
    ratios = compute_foreground_ratio(mask_dir)
    plt.figure(figsize=(5, 4))
    plt.hist(ratios * 100, bins=30, edgecolor="black")
    plt.xlabel("Foreground Ratio (%)")
    plt.ylabel("Number of Images")
    plt.title(f"Foreground Pixel Distribution ({dataset_name})")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "Foreground_Ratio_Histogram.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_params(lr, batch_size, loss_weight, fold_ids_list, dataset_full, tr_transform, ba_transform):
    dice_scores = []
    for fold, (train_ids, val_ids) in enumerate(fold_ids_list):
        train_dataset = DatasetInit(image_dir=IMG_DIR, mask_dir=MASK_DIR, transform=tr_transform)
        val_dataset = DatasetInit(image_dir=IMG_DIR, mask_dir=MASK_DIR, transform=ba_transform)
        train_subset = Subset(train_dataset, train_ids)
        val_subset = Subset(val_dataset, val_ids)
        train_loader = get_loaders(train_subset, batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)
        val_loader = get_loaders(val_subset, batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
        model = INFNet(3, 1).to(DEVICE)
        loss_fn = get_loss_fn(loss_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1, verbose=False)
        scaler = torch.cuda.amp.GradScaler()
        best_dice = 0.0
        for epoch in range(NUM_EPOCHS):
            train_fn(train_loader, val_loader, model, optimizer, loss_fn, scaler, epoch, amp=USE_AMP, scheduler=scheduler, dlr=True, device=DEVICE)
            _, _, dice, _, _, _, _, _ = val_fn(val_loader, model, device=DEVICE, folder=None, save_attention_map=False)
            if dice > best_dice:
                best_dice = dice
        dice_scores.append(best_dice)
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
    return mean(dice_scores)

def main():
    ba_transform = basic_transform(height=IMG_HEIGHT, width=IMG_WIDTH)
    tr_transform = train_transform(height=IMG_HEIGHT, width=IMG_WIDTH)
    dataset_full = DatasetInit(image_dir=IMG_DIR, mask_dir=MASK_DIR, transform=ba_transform)
    kf = KFold(n_splits=NUM_FOLD, shuffle=True, random_state=42)
    fold_ids_list = list(kf.split(dataset_full))

    if ENABLE_GRID_SEARCH:
        param_names = list(PARAM_GRID.keys())
        param_combinations = list(product(*PARAM_GRID.values()))
        results = []
        for i, combo in enumerate(param_combinations):
            params = dict(zip(param_names, combo))
            try:
                avg_dice = evaluate_params(
                    lr=params["learning_rate"],
                    batch_size=params["batch_size"],
                    loss_weight=params["loss_weight"],
                    fold_ids_list=fold_ids_list,
                    dataset_full=dataset_full,
                    tr_transform=tr_transform,
                    ba_transform=ba_transform
                )
                results.append((avg_dice, params))
            except Exception as e:
                results.append((0.0, params))
        results.sort(key=lambda x: x[0], reverse=True)
        best_dice, best_params = results[0]
        print("\n" + "="*50)
        print("🏆 Best Hyperparameters:")
        print(f"Dice: {best_dice:.4f}")
        for k, v in best_params.items():
            print(f"{k}: {v}")
        print("="*50)
        with open("grid_search_results.txt", "w") as f:
            f.write("Dice\tParams\n")
            for dice, params in results:
                f.write(f"{dice:.4f}\t{params}\n")

    quality_dir = "data_quality_analysis"
    os.makedirs(quality_dir, exist_ok=True)
    plot_foreground_boxplot(MASK_DIR, quality_dir, DATASET)
    plot_foreground_histogram(MASK_DIR, quality_dir, DATASET)
    visualize_image_mask_pairs(IMG_DIR, MASK_DIR, quality_dir, 3)

if __name__ == "__main__":
    main()