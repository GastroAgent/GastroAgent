import os
import shutil
import random
from pathlib import Path

# 配置路径
src_dir = "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/new_cropped-2004-2010"
val_dir = "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/val_new_cropped-2004-2010"  # 新验证集文件夹

# 创建验证文件夹
Path(val_dir).mkdir(parents=True, exist_ok=True)

# 支持的图像扩展名
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

# 获取所有图像文件
all_files = [
    f for f in Path(src_dir).iterdir()
    if f.is_file() and f.suffix.lower() in image_extensions
]

print(f"找到 {len(all_files)} 张图像")

# 设置随机种子以便可复现（可选）
random.seed(42)

# 随机选择 20% 作为验证集
val_ratio = 0.08
num_val = int(len(all_files) * val_ratio)
print(num_val)

val_files = random.sample(all_files, num_val)

print(f"将移动 {len(val_files)} 张图像到验证集")

# 移动文件（若要复制则用 shutil.copy2）
for f in val_files:
    shutil.move(str(f), os.path.join(val_dir, f.name))

print("✅ 完成！")