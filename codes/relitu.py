import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_heatmap(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # 将图像分割成5x5的部分
    h, w = image.shape
    block_size_h = h // 5
    block_size_w = w // 5
    heatmap_values = np.zeros((5, 5))
    
    for i in range(5):
        for j in range(5):
            block = image[i * block_size_h:(i + 1) * block_size_h, j * block_size_w:(j + 1) * block_size_w]
            heatmap_values[i, j] = np.mean(block)
    
    return heatmap_values

def plot_heatmaps(data, titles):
    num_algorithms = len(data)
    num_rows = 2  # 2行
    num_cols = 4  # 4列
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 6))
    axes = axes.flatten()
    
    # 为颜色条创建一个额外的轴
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for idx, (heatmap_values, title) in enumerate(zip(data, titles)):
        sns.heatmap(heatmap_values, ax=axes[idx], cbar=(idx == num_algorithms - 1), cbar_ax=None if idx != num_algorithms - 1 else cbar_ax, annot=True, fmt=".2f", cmap="RdPu")
        axes[idx].set_title(title, fontsize=14)
        avg = np.mean(heatmap_values)
        std = np.std(heatmap_values)
        axes[idx].set_xlabel(f"Avg: {avg:.2f} Std: {std:.2f}", fontsize=12)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    
    # 隐藏任何未使用的子图
    for i in range(len(data), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

# 图像文件路径列表
image_paths = [
    r"D:\Wz_Project_Learning\Super_Resolution_Reconstruction\dataset\H_data\AID-dataset\val\HR\bareland_54.png",
    r"D:\Wz_Project_Learning\Super_Resolution_Reconstruction\HAUNet_RSISR\experiment\SRDD\SRDDx2_AID\results\bareland_54.png",
    r"D:\Wz_Project_Learning\Super_Resolution_Reconstruction\HAUNet_RSISR\experiment\HSENet\HSENetx2_AID\results\bareland_54.png",
    r"D:\Wz_Project_Learning\Super_Resolution_Reconstruction\HAUNet_RSISR\experiment\TransENet\TransENetx2_AID\results\bareland_54.png",
    r"D:\Wz_Project_Learning\Super_Resolution_Reconstruction\HAUNet_RSISR\experiment\FENet\FENetx2_AID\results\bareland_54.png",
    r"D:\Wz_Project_Learning\Super_Resolution_Reconstruction\HAUNet_RSISR\experiment\LGCNet\LGCNetx2_AID\results\bareland_54.png",
    r"D:\Wz_Project_Learning\Super_Resolution_Reconstruction\HAUNet_RSISR\experiment\OMnisr\OMnisrx2_AID\results\bareland_54.png",
    r"D:\Wz_Project_Learning\Super_Resolution_Reconstruction\HAUNet_RSISR\experiment\FSMamba\FSMambax2_AID\results\bareland_54.png"
]

# 算法名称列表
algorithm_names = ["HR", "SRDD", "HSENet", "TransENet", "FENet", "LGCNet", "OMnisr", "FSMamba (ours)"]

# 计算每个图像的热力值
heatmaps = []
for path in image_paths:
    heatmap = calculate_heatmap(path)
    heatmaps.append(heatmap)

# 绘制热力图
plot_heatmaps(heatmaps, algorithm_names)

