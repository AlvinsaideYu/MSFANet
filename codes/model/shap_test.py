import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from fsmamba import EnhancedFSMamba
from option import args

# 加载单张图片并进行预处理
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),  # 假设输入图片为64x64
        transforms.ToTensor(),  # 转换为Tensor
    ])
    image = preprocess(image).unsqueeze(0)  # 添加batch维度
    return image

# 使用SHAP的DeepExplainer进行模型解释
def shap_explain_image(model, image):
    explainer = shap.DeepExplainer(model, image)
    shap_values = explainer.shap_values(image)
    return shap_values

# 可视化原始图片、模型输出和SHAP解释
def visualize_results(original_image, output_image, shap_values):
    # 将Tensor转换为Numpy格式
    original_image_np = original_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    output_image_np = output_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # 绘制图像
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 显示原始图片
    axs[0].imshow(original_image_np)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # 显示超分辨率结果
    axs[1].imshow(output_image_np)
    axs[1].set_title('Super-Resolution Output')
    axs[1].axis('off')

    # 显示SHAP解释结果
    shap.image_plot(shap_values, original_image.cpu().numpy(), ax=axs[2])

    # 设置SHAP解释的标题
    axs[2].set_title('SHAP Explanation')
    plt.show()

# 主函数
if __name__ == "__main__":
    # 加载FSMamba模型
    model = EnhancedFSMamba(args).cuda()
    model.eval()  # 设置为评估模式

    # 加载单张图片
    image_path = "church_39.png"  # 替换为你图片的路径
    input_image = load_image(image_path).cuda()

    # 使用模型生成超分辨率图像
    with torch.no_grad():
        output_image = model(input_image)

    # 使用SHAP解释
    shap_values = shap_explain_image(model, input_image)

    # 可视化原始图片、超分辨率结果和SHAP解释
    visualize_results(input_image, output_image, shap_values)
