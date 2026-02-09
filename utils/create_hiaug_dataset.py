# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import json
# import numpy as np
# import random
# import pickle
# import torch
# import torchvision.transforms as transforms
# from torchvision.transforms import ToPILImage
# from PIL import Image
#
# inverse_normalize = transforms.Normalize(
#     mean=[-1, -1, -1],  # 计算: -mean/std = -0.5/0.5 = -1
#     std=[2, 2, 2]       # 计算: 1/std = 1/0.5 = 2
# )
# to_pil = ToPILImage()
#
# # 3. 逆归一化并保存图像的函数
# def save_normalized_tensor(tensor_image, filename):
#     """
#     Args:
#         tensor_image (torch.Tensor): 归一化后的图像 Tensor，形状为 [C, H, W]。
#         filename (str): 保存图像的文件路径。
#     """
#     tensor_image = tensor_image.cpu()
#     # 注意: transforms.Normalize 期望 batch 维度或直接是 [C, H, W]
#     inv_tensor = inverse_normalize(tensor_image)
#     pil_image = to_pil(inv_tensor)
#     pil_image.save(filename)
#     print(f"图像已保存至: {filename}")
# class AddLocalGaussianNoise(object):
#     def __init__(self, mean=0., std=0.05, probability=0.25, local_radius=36):
#         self.mean = mean
#         self.std = std
#         self.probability = probability
#         self.local_radius = local_radius # 控制噪声应用的局部区域大小
#
#     def __call__(self, img):
#         if torch.rand(1) < self.probability:
#             # 转换为numpy数组以便操作
#             img_np = np.array(img)
#             # 选择一个随机点作为噪声中心
#             height, width = img_np.shape[:2]
#             center_x = np.random.randint(self.local_radius, width - self.local_radius)
#             center_y = np.random.randint(self.local_radius, height - self.local_radius)
#             # 在选定中心周围生成一个高斯噪声块
#             y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
#             mask = x * x + y * y <= self.local_radius * self.local_radius
#             # 生成高斯噪声
#             noise = np.zeros_like(img_np)
#             noise[mask] = np.random.normal(self.mean, self.std, size=(mask.sum(), 3)) if len(img_np.shape) == 3 else np.random.normal(self.mean, self.std, size=(mask.sum()))
#             # 应用噪声
#             img_np[mask] += noise[mask]
#             # 确保像素值仍在有效范围内
#             img_np = np.clip(img_np, 0, 255).astype(np.uint8)
#             # 返回PIL图像
#             return Image.fromarray(img_np)
#         return img
#
#
#
# dataset = json.load(open('/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/diffusion_dit_dataset_sup_tiny.json', 'r'))
#
# base_transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
#
# # 轻微扰动保持图像结构
# light_transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)),  # 微小变形
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),  # 轻柔颜色变化
#     transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 0.3))], p=1),
#     AddLocalGaussianNoise(mean=0.1, std=0.3, probability=1, local_radius=36),
#     # transforms.RandomApply([transforms.Grayscale(3)], p=0.3),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
#
# # 平衡增强策略
# mid_transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.RandomAffine(degrees=4, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=(-5, 5)),  # 明显但可控的变形
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.2),  # 显著颜色变化
#     transforms.RandomApply([transforms.GaussianBlur(7, sigma=(0.5, 1.0))], p=1),
#     AddLocalGaussianNoise(mean=0.1, std=0.2, probability=1, local_radius=36),
#     transforms.RandomAdjustSharpness(sharpness_factor=2, p=1),  # 新增锐度调整
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
#
# # 强烈增强策略
# agg_transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.RandomAffine(degrees=5, translate=(0.2, 0.2), scale=(0.75, 1.25), shear=(-7, 7)),  # 明显但可控的变形
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.2),  # 显著颜色变化
#     transforms.RandomApply([transforms.GaussianBlur(7, sigma=(0.5, 1.0))], p=1),
#     transforms.RandomApply([transforms.Grayscale(3)], p=1),
#     AddLocalGaussianNoise(mean=0.2, std=0.3, probability=1, local_radius=36),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
#
# light_prompt = '对图像进行细微的仿射变换和颜色调整，以保留图像的基本结构同时引入轻微的变化。应用轻度模糊和细粒噪声，使图像产生不易察觉的差异。'
# mid_prompt = '图像经过适度的变形、显著的颜色与锐度调整，以及中等程度的模糊处理。通过添加适量的局部噪声来增加图像的多样性，同时保持整体视觉效果的平衡。'
# agg_prompt = '对图像施加强烈的变形、颜色变化和灰度转换，结合较强的模糊效果和明显的噪声添加，大幅改变原始图像外观。'
# new_dataset = []
# label_dis2id = {}
# label_id = 0
# for data in dataset:
#     image_path = data['image']
#     disease = data['disease']
#     if disease in label_dis2id:
#         pass
#     else:
#         label_dis2id[disease] = label_id
#         label_id += 1
#
#     x0 = base_transform(Image.open(image_path).convert('RGB'))
#     for idx, prompt, transform in ((1, light_prompt, light_transform), (2, mid_prompt, mid_transform), (3, agg_prompt, agg_transform)):
#         new_data = {}
#         new_data['x0'] = x0
#         new_data['prompt'] = prompt
#         new_data['x1'] = transform(Image.open(image_path).convert('RGB'))
#         new_data['y'] = label_dis2id[disease]
#         new_data['disease'] = disease
#         save_normalized_tensor(new_data['x1'],
#                                f'/home/lab/work/Diffusion/flow-based-models/aug_images/{disease}_hiaug_{idx}.jpg')
#         new_dataset.append(new_data)
#
# with open('/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/diffusion_dataset_sup_tiny_hiaug.pkl', 'wb') as f:  # 注意模式是 'wb'
#     pickle.dump(new_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)  # 使用最高协议以获得更好的性能和兼容性
#
# with open('/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/diffusion_label2id_sup_tiny_hiaug.json', 'w') as f:
#     json.dump(label_dis2id, f, indent=4, ensure_ascii=False)
#
#
#
########################################################################################################
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import numpy as np
import random
import pickle
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
from glob import glob

label_dis2id = {}
label_id = 0
dataset = []

image_folder = '/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/book_sim_data'
diseases_folder = os.listdir(image_folder)

for disease in diseases_folder:
    if disease not in label_dis2id:
        label_dis2id[disease] = label_id
        label_id += 1
    image_paths = os.path.join(image_folder, disease)
    images = glob(os.path.join(image_paths,'pos_*'))
    
    image_x0 = images[0]
    for image in images[1:]:
        image_x1 = image
        caption = f'生成疾病（{disease}）。'
        # caption = f'生成同种疾病（{disease}）的其他图像。'
        dataset.append({
            'x0': image_x0,
            'x1': image_x1,
            'caption': caption,
            'label': label_dis2id[disease]
        })

# image_folder = '/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/processed_sim_data'
# diseases_folder = os.listdir(image_folder)
#
# for disease in diseases_folder:
#     if disease not in label_dis2id:
#         label_dis2id[disease] = label_id
#         label_id += 1
#     image_paths = os.path.join(image_folder, disease)
#     images = glob(os.path.join(image_paths,'pos_*'))
#     if len(images) < 2:
#         continue
#     image_x0 = images[0]
#     for image in images[1:]:
#         image_x1 = image
#         caption = f'生成同种疾病（{disease}）的其他图像。'
#         dataset.append({
#             'x0': image_x0,
#             'x1': image_x1,
#             'caption': caption,
#             'label': label_dis2id[disease]
#         })

with open('/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/diffusion_dataset_sup_tiny_otherdis.json',
          'w') as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)
print('Dataset Size: ', len(dataset))
with open('/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/diffusion_label2id_sup_tiny_hidis.json',
          'w') as f:
    json.dump(label_dis2id, f, indent=4, ensure_ascii=False)
print('Labels Size: ', len(label_dis2id))


