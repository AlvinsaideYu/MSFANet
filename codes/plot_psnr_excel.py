import re
import pandas as pd

# 读取txt文件
with open('D:\Wz_Project_Learning\Super_Resolution_Reconstruction\HAUNet_RSISR\Ablation_experiment\FSMambax4_UCMerced\log.txt', 'r') as file:
    data = file.readlines()

# 初始化空列表存储psnr和ssim
psnr_list = []
# ssim_list = []

# 正则表达式匹配psnr和ssim信息
for line in data:
    psnr_match = re.search(r'psnr:\s*([0-9.]+)', line)
    # ssim_match = re.search(r'ssim:\s*([0-9.]+)', line)
    
    # if psnr_match and ssim_match:
    if psnr_match:
        psnr = float(psnr_match.group(1))
        # ssim = float(ssim_match.group(1))
        psnr_list.append(psnr)
        # ssim_list.append(ssim)

# 创建DataFrame
df = pd.DataFrame({
    'PSNR': psnr_list,
    # 'SSIM': ssim_list
})

# 写入到Excel文件
df.to_excel('output.xlsx', index=False)

print("信息已成功提取并写入到output.xlsx文件中")
