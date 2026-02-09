# 验证依赖库
import torch
import torchvision
#import timm
import cv2
import numpy
print("PyTorch 版本:", torch.__version__)  # 需 >=2.0.0

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于显示进度条
import glob

INPUT_root = "/mnt/inaisfs/data/home/tansy_criait/Datasets/mini-imagenet-folder"
OUTPUT_root = "/mnt/inaisfs/data/home/tansy_criait/Datasets/depth/mini-imagenet-folder"
INPUT_FOLDERs = glob.glob(f"{INPUT_root}/*")
# INPUT_FOLDERs = ["/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/data/胃/eval/正常胃窦"]

# -------------------------- 配置参数（修改为你的文件夹路径） --------------------------
PROJECT_PATH = "/mnt/inaisfs/data/home/tansy_criait/abnormal_dectect/Depth-Anything-V2-main/Depth-Anything-V2-main"  # 项目根目录
MODEL_TYPE = "vitb"  # 模型类型
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # 支持的图片格式
MAX_RECURSION_DEPTH = 0  # 最大递归目录深度，设置为0表示只处理当前目录，-1表示无限制
# ---------------------------------------------------------------------------------------

# 修复模块导入（强制添加项目路径）
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
from depth_anything_v2.dpt import DepthAnythingV2

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 初始化设备和模型
DEVICE = "cuda"
model_configs = {
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]}
}
CHECKPOINT_PATH = os.path.join(PROJECT_PATH, f"checkpoints/depth_anything_v2_{MODEL_TYPE}.pth")

# 加载模型
model = DepthAnythingV2(**model_configs[MODEL_TYPE])
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()
print("模型加载成功，开始批量处理...")

def get_image_files_recursive(root_dir, current_depth=0):
    """
    递归获取所有图片文件及其相对路径
    返回格式: [(绝对路径, 相对路径), ...]
    """
    image_files = []
    
    # 检查是否超过最大递归深度
    if MAX_RECURSION_DEPTH != -1 and current_depth > MAX_RECURSION_DEPTH:
        return image_files
    
    # 遍历当前目录
    for entry in os.scandir(root_dir):
        if entry.is_dir(follow_symlinks=False):
            # 递归处理子目录，深度+1
            subdir_files = get_image_files_recursive(
                entry.path, 
                current_depth + 1
            )
            image_files.extend(subdir_files)
        elif entry.is_file() and entry.name.lower().endswith(SUPPORTED_FORMATS):
            # 计算相对路径（相对于INPUT_FOLDER）
            rel_path = os.path.relpath(entry.path, INPUT_FOLDER)
            image_files.append((entry.path, rel_path))
    
    return image_files

for INPUT_FOLDER in INPUT_FOLDERs:
    print(INPUT_FOLDER)
    OUTPUT_FOLDER = INPUT_FOLDER.replace(INPUT_root, OUTPUT_root)
    # 创建输出文件夹（如果不存在）
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"输出文件夹已准备: {OUTPUT_FOLDER}")
    # 获取所有图片文件（包括子目录）
    image_files = get_image_files_recursive(INPUT_FOLDER)

    if not image_files:
        raise FileNotFoundError(f"输入文件夹及其子目录中未找到任何图片文件: {INPUT_FOLDER}\n支持格式: {SUPPORTED_FORMATS}")

    print(f"共发现 {len(image_files)} 个图片文件，开始处理...")

    # 批量处理图片
    for abs_path, rel_path in tqdm(image_files, desc="批量处理进度"):
        try:
            # 构建输出路径，保持目录结构
            output_file_path = os.path.join(OUTPUT_FOLDER, rel_path)
            # 创建输出文件所在的目录
            output_dir = os.path.dirname(output_file_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 提取文件名（不含扩展名）用于保存深度图
            name_without_ext = os.path.splitext(output_file_path)[0]
            output_path = f"{name_without_ext}_depth.png"

            # 读取图片（支持中文路径）
            raw_img = cv2.imdecode(np.fromfile(abs_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw_img is None:
                print(f"警告：无法读取文件 {abs_path}，已跳过")
                continue

            # 深度估计
            with torch.no_grad():
                depth_map = model.infer_image(raw_img)

            # 处理并保存深度图
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_8bit = (depth_normalized * 255).astype(np.uint8)
            
            # 保存深度图（支持中文路径）
            is_success, im_buf_arr = cv2.imencode(".png", depth_8bit)
            if is_success:
                im_buf_arr.tofile(output_path)
            else:
                print(f"警告：无法保存文件 {output_path}，已跳过")
                continue

        except Exception as e:
            print(f"处理文件 {abs_path} 时出错: {str(e)}，已跳过")
            continue

    print(f"批量处理完成！共处理 {len(image_files)} 张图片，结果保存在: {OUTPUT_FOLDER}")
