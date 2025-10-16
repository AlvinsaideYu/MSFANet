import sys
import os

# 添加项目根目录到sys.path中
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch
from model.fsmamba import EnhancedFSMamba  # 确保导入正确的模型文件

class Args:
    n_colors = 3  # 设定您的模型参数

args = Args()

# 初始化模型
model = EnhancedFSMamba(args)
model.eval()  # 切换模型到评估模式

# 创建一个示例输入
dummy_input = torch.randn(1, 3, 48, 48)  # 根据您的输入大小调整

# 导出为 ONNX 格式
torch.onnx.export(model, dummy_input, "fsmamba_net.onnx", 
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

print("ONNX 模型已成功导出为 fsmamba_net.onnx")
