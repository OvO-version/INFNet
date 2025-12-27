# 🩺 INFNet：基于结构相关性信息融合的多源医学图像分割方法研究

INFNet 是一种新型医学图像分割框架，通过引入结构感知瓶颈交互，在多种成像模态（如超声、CT、眼底彩照等）和解剖结构（如视网膜血管、乳腺肿块、肝脏等）上实现领先的分割性能。

> ✨ **核心优势**：
> - 对小目标与模糊边界具有强鲁棒性  
> - 支持 2D/3D、单类/多类统一架构  
> - 模块化设计  

---

## 📊 支持的数据集

我们在 8 个公开医学图像分割数据集 上验证了 INFNet 的有效性，涵盖视网膜、皮肤、牙齿、乳腺、肝脏及腹部多器官任务。

| 简称     | 部位   | 目标           | 成像模态       | 样本数量 | 数据来源 |
|----------|--------|----------------|----------------|----------|----------|
| DRIVE    | 头部   | 视网膜血管     | 眼底彩照       | 40       | [链接](https://drive.grand-challenge.org/) |
| CHASEDB1 | 头部   | 视网膜血管     | 眼底彩照       | 28       | [链接](https://www.kaggle.com/datasets/khoongweihao/chasedb1) |
| STARE    | 头部   | 视网膜血管     | 眼底彩照       | 20       | [链接](https://cecas.clemson.edu/~ahoover/stare/) |
| SLD      | 皮肤   | 皮肤病变       | 皮肤镜图像     | 200      | [链接](https://challenge.isic-archive.com/data/) |
| TOOTH    | 头部   | 牙齿结构       | 锥形束 CT      | 1,998    | [链接](https://tianchi.aliyun.com/dataset/156596) |
| CTL      | 腹部   | 肝脏           | CT             | 116      | [链接](https://www.kaggle.com/datasets/siatsyx/ct2us) |
| BUSI     | 胸部   | 乳腺肿块       | 超声图像       | 210      | [链接](https://www.kaggle.com/datasets/anaselmasry/datasetbusiwithgt) |
| SYNAPSE  | 腹部   | 多器官（8类）  | CT             | 2,212    | [链接](http://medicaldecathlon.com/) |

> 💡 注意：请从上述链接手动下载数据，并按下方说明组织目录结构。

---

## 📁 项目结构

```bash
INFNet/
├── data/                   # 数据集根目录（需手动放置）
├── train.py               # 训练脚本
├── output
├── visualize_results.py   # 生成实验图表（Dice 箱线图、前景比例分布等）
└── README.md

---
##如需完整的训练代码（包括数据预处理、损失函数、日志记录等），欢迎通过 GitHub Issue 联系我！
