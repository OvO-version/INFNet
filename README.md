🩺 INFNet：基于结构相关性信息融合的多源医学图像分割方法研究





INFNet 是一种新型医学图像分割框架，通过引入结构感知瓶颈交互，在多种成像模态（如超声、CT、眼底彩照等）和解剖结构（如视网膜血管、乳腺肿块、肝脏等）上实现领先的分割性能。

✨ 核心优势：

对小目标与模糊边界具有强鲁棒性
支持 2D/3D、单类/多类统一架构
模块化设计
📊 支持的数据集
我们在 8 个公开医学图像分割数据集 上验证了 INFNet 的有效性，涵盖视网膜、皮肤、牙齿、乳腺、肝脏及腹部多器官任务。

简称	部位	目标	成像模态	样本数量	数据来源
DRIVE	头部	视网膜血管	眼底彩照	40	链接
CHASEDB1	头部	视网膜血管	眼底彩照	28	链接
STARE	头部	视网膜血管	眼底彩照	20	链接
SLD	皮肤	皮肤病变	皮肤镜图像	200	链接
TOOTH	头部	牙齿结构	锥形束 CT	1,998	链接
CTL	腹部	肝脏	CT	116	链接
BUSI	胸部	乳腺肿块	超声图像	210	链接
SYNAPSE	腹部	多器官（8类）	CT	2,212	链接
💡 注意：请从上述链接手动下载数据，并按下方说明组织目录结构。

📁 项目结构
bash
编辑
INFNet/
├── data/                   # 数据集根目录（需手动放置）
├── models/
│   └── infnet.py          # INFNet 主模型代码
├── train.py               # 训练脚本
├── test.py                # 测试与评估脚本
├── visualize_results.py   # 生成实验图表（Dice 箱线图、前景比例分布等）
├── utils/
│   ├── dataset.py         # 自定义数据加载器
│   └── metrics.py         # Dice、HD95 等指标计算
├── configs/               # 超参数配置（学习率、batch size、损失权重等）
└── README.md
⚙️ 快速开始
1. 环境配置
bash
编辑
conda create -n infnet python=3.9
conda activate infnet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2. 数据集准备
将各数据集按以下格式放入 ./data/ 目录：

text
编辑
data/
├── drive/
│   ├── images/    # 原始图像
│   └── masks/     # 分割标签
└── ...

3. 启动训练
bash
编辑
python train.py --dataset busi --lr 1e-4 --batch_size 4 --loss_weight 0.5
常用参数说明：

--dataset：指定数据集（如 busi, drive, synapse）
--lr：学习率（支持 1e-5, 1e-4, 5e-4）
--batch_size：批大小（2, 4, 8）
--loss_weight：Dice+BCE 损失中 Dice 的权重（如 0.5 表示 0.5:0.5）
4. 生成实验图表
运行以下命令可自动生成论文所需分析图：

bash
编辑
python visualize_results.py --dataset busi --output_dir ./figures/
其他图是由visio或origin做的
📈 实验结果
在最优配置（学习率=1e⁻⁴，Batch Size=4，Dice+BCE 损失权重=0.5:0.5）下，INFNet 在多个数据集上取得如下 Dice 分数：

数据集	Dice (%)
DRIVE	63.46
TOOTH	91.55
CTL	95.13
BUSI	68.77
CHASEDB1	57.24
STARE	48.47
完整实验结果与消融分析详见论文。

📜 许可证
本项目采用 MIT 许可证 开源。
