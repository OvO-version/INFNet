import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# ================= 配置 =================
IMG_DIR = r"E:\DLPrograms\Project_ZYTWYX\Dataset_BUSI_with_GT\image"
SAVE_DIR = r"../../RecordData/JQXXJC_BUSI/data_quality"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_SAMPLES = 40  # 抽样图像数（够用即可）

# ================= 主逻辑 =================
def plot_pixel_distribution_before_after():
    image_names = random.sample(os.listdir(IMG_DIR), NUM_SAMPLES)

    pixels_before = []
    pixels_after = []

    for name in image_names:
        img = cv2.imread(os.path.join(IMG_DIR, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = img.astype(np.float32)
        pixels_before.append(img.flatten())

        # 单图归一化（与你论文描述一致）
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        pixels_after.append(img_norm.flatten())

    pixels_before = np.concatenate(pixels_before)
    pixels_after = np.concatenate(pixels_after)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(pixels_before, bins=50, color="gray")
    plt.title("Before Normalization")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(pixels_after, bins=50, color="gray")
    plt.title("After Normalization")
    plt.xlabel("Normalized Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "Pixel_Distribution_Before_After_Normalization.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[OK] Saved Figure A to {save_path}")


if __name__ == "__main__":
    plot_pixel_distribution_before_after()
