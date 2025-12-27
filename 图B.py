import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# ================= 配置 =================
IMG_DIR = r"E:\DLPrograms\Project_ZYTWYX\Dataset_BUSI_with_GT\image"
SAVE_DIR = r"../../RecordData/JQXXJC_BUSI/data_quality"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_SIZE = (224, 224)

# ================= 主逻辑 =================
def visualize_preprocessing_effect():
    name = random.choice(os.listdir(IMG_DIR))
    img = cv2.imread(os.path.join(IMG_DIR, name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    # normalization（可视化用 min-max）
    img_norm = img_resized.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_norm)
    plt.title("After Preprocessing")
    plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "Preprocessing_Before_After_Visualization.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[OK] Saved Figure B to {save_path}")


if __name__ == "__main__":
    visualize_preprocessing_effect()
