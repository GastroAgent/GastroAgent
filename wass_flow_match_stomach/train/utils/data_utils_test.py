import json

from .data_loader_hiaug import *
from .data_loader_test import *
import os
import shutil
##########################添加新的训练数据###################################
def create_dataset(
        data_path_A='/mnt/inaisfs/data/home/tansy_criait/flow_match/test_data/食管_Cropped/A级',
        data_path_B='/mnt/inaisfs/data/home/tansy_criait/flow_match/test_data/食管_Cropped/C级',
        label_A='食管炎-A级', label_B='食管炎-C级', y=1, shuffle_AB=False, caption_dict=None,
        transform=None, transform_A=None, transform_B=None, batchsize=4, return_dataset=False
    ):
    caption = None
    if caption_dict is not None:
        if isinstance(caption_dict, dict):
            caption = caption_dict[(label_A, label_B)]
        elif isinstance(caption_dict, str):
            caption = caption_dict
    else:
        caption = None
    dataset = MedicalA2BDataset(
        data_path_A=data_path_A,
        data_path_B=data_path_B,
        caption=caption,
        label_A=label_A, label_B=label_B, y=y, shuffle_AB=shuffle_AB,
        transform=transform, transform_A=transform_A, transform_B=transform_B
    )
    print('Dataset Size: ', len(dataset))
    print(f'{label_A} -> {label_B}')
    print('-'*100)
    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    if return_dataset:
        return dataloader, dataset
    return dataloader

def get_all_pairs(lst, dia=True):
    pairs = []
    n = len(lst)
    if dia:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((lst[i], lst[j]))
    else:
        for i in range(n):
            for j in range(n):
                pairs.append((lst[i], lst[j]))

    return pairs

def sort_files_by_direction(source_folder, left_keyword='left', right_keyword='right'):
    """
    将指定文件夹中的文件根据文件名中的关键词分类，
    并移动到与源文件夹同级的 left_folder 和 right_folder 中。

    参数:
    source_folder (str): 包含待分类文件的源文件夹路径。
    left_keyword (str): 识别“左侧”文件的关键词，默认 'left'。
    right_keyword (str): 识别“右侧”文件的关键词，默认 'right'。
    """
    # 获取源文件夹的父目录
    parent_folder = os.path.dirname(os.path.abspath(source_folder))
    source_file_name = source_folder.split('/')[-1]
    # 定义同级的目标文件夹路径
    left_folder = os.path.join(parent_folder, f'{source_file_name}_left_folder')
    right_folder = os.path.join(parent_folder, f'{source_file_name}_right_folder')

    # 确保目标文件夹存在
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)

        # 只处理文件（忽略子目录）
        if os.path.isfile(source_file):
            target_path = None
            if left_keyword in filename.lower():  # 使用 lower() 实现大小写不敏感
                target_path = os.path.join(left_folder, filename)
            elif right_keyword in filename.lower():
                target_path = os.path.join(right_folder, filename)

            # 如果匹配关键词，则移动文件
            if target_path:
                # 处理重名文件：添加数字后缀
                base_name, ext = os.path.splitext(target_path)
                counter = 1
                final_target_path = target_path
                while os.path.exists(final_target_path):
                    final_target_path = f"{base_name}_{counter}{ext}"
                    counter += 1
                shutil.move(source_file, final_target_path)
                print(f"Moved {filename} to {final_target_path}")

    print("文件分类完成。")

def create_dataloaders(image_dir=None, mode='full', transform=None, transform_A=None, transform_B=None, batchsize=4):
    if image_dir is not None:
        raise NotImplementedError('')
    ##########################添加新的训练数据###################################
    dataloaders = []
    dataloader = create_dataset(
        data_path_A='/mnt/inaisfs/data/home/tansy_criait/flow_match/test_data/食管_Cropped2/A级',
        data_path_B='/mnt/inaisfs/data/home/tansy_criait/flow_match/test_data/食管_Cropped2/C级',
        label_A='食管炎-A级', label_B='食管炎-C级', y=2, shuffle_AB=True, batchsize=batchsize,
        transform=transform, transform_A=transform, transform_B=transform)

    dataloaders.append(dataloader)
    return dataloaders

def create_dataloaders_by_pairs(image_dir: str=None, pairs: tuple[str, str]=None, label_start = 0 , caption_map: dict = None, caption_dict: dict = None,
                                shuffle_AB=True, transform=None, transform_A=None, transform_B=None, batchsize=4, return_dataset=False, **kwargs):
    if image_dir is None:
        raise NotImplementedError('')
    datasets = []
    try:
        label_map = json.load(open('/mnt/inaisfs/data/home/tansy_criait/flow_match/utils/label_map.json', 'r'))
        label_start = max(list(label_map.values())) + 1
    except:
        label_map = {}
    caption_map = caption_map
    dataloaders = []
    for pair in pairs:
        if pair[1] not in label_map:
            label_map[pair[1]] = label_start
            label_start += 1
        y = label_map[pair[1]]
        label_A = pair[0]
        label_B = pair[1]
        data_path_A = os.path.join(image_dir, label_A)
        data_path_B = os.path.join(image_dir, label_B)

        if label_A not in caption_map:
            caption_map[label_A] = label_A
        if label_B not in caption_map:
            caption_map[label_B] = label_B
        dataloader = create_dataset(
            data_path_A=data_path_A,
            data_path_B=data_path_B,
            label_A=caption_map[label_A], label_B=caption_map[label_B],
            caption_dict=caption_dict, return_dataset=return_dataset,
            y=y, shuffle_AB=shuffle_AB, batchsize=batchsize,
            transform=transform, transform_A=transform_A, transform_B=transform_B)
        if return_dataset:
            dataset = dataloader[1]
            datasets.append(dataset)
            dataloader = dataloader[0]

        dataloaders.append(dataloader)
    json.dump(label_map, open('/mnt/inaisfs/data/home/tansy_criait/flow_match/utils/label_map.json', 'w'),
              indent=4, ensure_ascii=False)
    if return_dataset:
        return dataloaders, datasets
    return dataloaders


if __name__ == '__main__':
    # sort_files_by_direction(
    #     '/mnt/inaisfs/data/home/tansy_criait/flow_match/test_data/Diabetic_Retinopathy_Arranged_datasets/4')
    folders_lst = os.listdir('/mnt/inaisfs/data/home/tansy_criait/flow_match/test_data/食管_cropped_update')

    pairs = get_all_pairs(folders_lst)
    print(pairs)
    caption_map = {
        '巴雷特食管': '巴雷特食管',
        '正常食管': '正常食管',
        '食管炎C级': '食管炎C级',
        '食管癌': '食管癌',
        '食管癌术后': '食管癌术后'
    }
    dataloaders = create_dataloaders_by_pairs(
        image_dir='/mnt/inaisfs/data/home/tansy_criait/flow_match/test_data/食管_cropped_update',
        pairs=pairs, caption_map=caption_map, caption_dict=None,
        label_start=4, shuffle_AB=True, transform=None, transform_A=None, transform_B=None,
    )

    print('Dataloader Size: ', len(dataloaders))



