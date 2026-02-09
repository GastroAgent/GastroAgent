import os
from tqdm import tqdm
import cv2
import numpy as np
from glob import glob

def crop_save(path, save_path='./cropped', faild_dir=''):
    image = cv2.imread(path)
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # HSV空间 只用于 Mask。
    except cv2.error:
        print(path)
        if faild_dir:
            file_name = path.split('/')[-1]
            cv2.imwrite(os.path.join(faild_dir, file_name), image)
        return path

    # 设定颜色范围（根据实际情况调整）
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([180, 255, 50])  # 这里假设亮度低于50的区域为黑色
    # 创建掩码
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # 反转掩码，使得非黑色区域为白色
    mask = cv2.bitwise_not(mask)

    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # 找到最大的轮廓
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width, _ = image.shape
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # 获取包围框
        x, y, w, h = cv2.boundingRect(largest_contour)
        # 裁剪图像
        bottom = min(y + 6 + h, height)
        top = max(y - 6, 0)
        left = max(x - 8, 0)
        right = min(x + w + 8, width)
        try:
            cropped_image = image[top:bottom, left:right]
            height, width, _ = cropped_image.shape
            if height < 28 or width < 28:
                if faild_dir:
                    file_name = path.split('/')[-1]
                    cv2.imwrite(os.path.join(faild_dir, file_name), image)
                    return path
            cv2.imwrite(save_path, cropped_image)
        except:
            cropped_image = image[y:y + h, x:x + w]
            height, width, _ = cropped_image.shape
            if height < 28 or width < 28:
                if faild_dir:
                    file_name = path.split('/')[-1]
                    cv2.imwrite(os.path.join(faild_dir, file_name), image)
                    return path
            cv2.imwrite(save_path, cropped_image)
        return ""
    else:
        print("No red region found.")
        print(path)
        if faild_dir:
            file_name = path.split('/')[-1]
            cv2.imwrite(os.path.join(faild_dir, file_name), image)
        return path

images = glob('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/our_data/new_cropped-2004-2010-endovit/*.jpg')
num_failed = 0
num_total = 0
file_idx = 1
for image in tqdm(images):
    num_total += 1
    
    new_path = os.path.join(f'/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/new_cropped-2004-2010',
                            image.split('/')[-1])
    if os.path.exists(new_path):
        continue
        
    failed = crop_save(image,
                       save_path=new_path, # new_path: 新存放路径。image: 旧的路径。
                       faild_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/Crop_Failed')
    if failed:
        num_failed += 1
        print(failed)
print('Total cropped images: {}'.format(num_total))
print('Failed cropped images: {}'.format(num_failed))
print('Done.')