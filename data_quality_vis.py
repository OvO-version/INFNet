import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

# ================= 基本配置 =================
DATASET = "BUSI"

IMG_DIR = r"E:\DLPrograms\Project_ZYTWYX\Dataset_BUSI_with_GT\image"
MASK_DIR = r"E:\DLPrograms\Project_ZYTWYX\Dataset_BUSI_with_GT\mask"

RESULT_ROOT = r"../../RecordData/JQXXJC_BUSI"
SAVE_DIR = os.path.join(RESULT_ROOT, "data_quality")

NUM_QUALITY_SAMPLES = 3

os.makedirs(SAVE_DIR, exist_ok=True)


# ================= 工具函数 =================
def list_image_files(dir_path):
    """只返回图像文件，避免误读"""
    return sorted([
        f for f in os.listdir(dir_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])


# ================= 前景比例统计 =================
def compute_foreground_ratio(mask_dir):
    ratios = []
    mask_files = list_image_files(mask_dir)

    for name in mask_files:
        mask_path = os.path.join(mask_dir, name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        fg_ratio = np.sum(mask > 0) / mask.size
        ratios.append(fg_ratio)

    return np.array(ratios)


# ================= 图 1：箱线图 =================
def plot_foreground_boxplot():
    ratios = compute_foreground_ratio(MASK_DIR)

    plt.figure(figsize=(5, 4))
    plt.boxplot([ratios], labels=[DATASET], showfliers=True)
    plt.ylabel("Foreground Pixel Ratio")
    plt.title("Foreground Pixel Ratio Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "Fig1_Foreground_Ratio_Boxplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")


# ================= 图 2：直方图 =================
def plot_foreground_histogram():
    ratios = compute_foreground_ratio(MASK_DIR)

    plt.figure(figsize=(5, 4))
    plt.hist(ratios * 100, bins=30, edgecolor="black")
    plt.xlabel("Foreground Ratio (%)")
    plt.ylabel("Number of Images")
    plt.title(f"Foreground Pixel Distribution ({DATASET})")
    plt.grid(alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "Foreground_Ratio_Histogram.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")


# ================= 图 3：图像-标注对 =================
def visualize_image_mask_pairs():
    img_files = list_image_files(IMG_DIR)
    num_samples = min(NUM_QUALITY_SAMPLES, len(img_files))
    samples = random.sample(img_files, num_samples)

    for i, name in enumerate(samples):
        img_path = os.path.join(IMG_DIR, name)
        mask_path = os.path.join(MASK_DIR, name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, f"Quality_ImageMask_{i}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[Saved] {save_path}")


# ================= 图 4：成像伪影示例 =================
def visualize_artifacts():
    img_files = list_image_files(IMG_DIR)
    num_samples = min(NUM_QUALITY_SAMPLES, len(img_files))
    samples = random.sample(img_files, num_samples)

    plt.figure(figsize=(9, 3))
    for i, name in enumerate(samples):
        img = cv2.imread(os.path.join(IMG_DIR, name))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title("Artifact Example")
        plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "Artifact_Examples.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")


# ================= 主入口 =================
if __name__ == "__main__":
    print("[INFO] Running data quality visualization...")

    plot_foreground_boxplot()
    plot_foreground_histogram()
    visualize_image_mask_pairs()
    visualize_artifacts()

    print("[DONE] All data quality figures have been generated.")
