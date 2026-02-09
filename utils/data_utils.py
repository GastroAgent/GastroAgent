from data_loader_hiaug import *

##########################添加新的训练数据###################################
def create_dataset(
        data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/A级',
        data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/C级',
        label_A='食管炎-A级', label_B='食管炎-C级', y=1, shuffle_AB=False,
        transform=None, transform_A=None, transform_B=None, batchsize=4,
):
    dataset = MedicalCLIPTinyA2BDataset(
        data_path_A=data_path_A,
        data_path_B=data_path_B,
        label_A=label_A, label_B=label_B, y=y, shuffle_AB=shuffle_AB,
        transform=transform, transform_A=transform_A, transform_B=transform_B
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    return dataloader

def create_dataloaders(image_dir=None, mode='full', transform=None, transform_A=None, transform_B=None, shuffle_AB=True, batchsize=4):
    # if image_dir is None:
    #     raise NotImplementedError('')
    ##########################添加新的训练数据###################################
    dataloaders = []
    dataloader = create_dataset(
        data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped2/A级',
        data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped2/C级',
        label_A='食管炎-A级', label_B='食管炎-C级', y=2, shuffle_AB=shuffle_AB, batchsize=batchsize,
        transform=transform, transform_A=transform, transform_B=transform)
    dataloaders.append(dataloader)
    
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped2/A级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped2/C级',
    #     label_A='食管炎-A级', label_B='食管炎-C级', y=2, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/A级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/C级',
    #     label_A='食管炎-A级', label_B='食管炎-C级', y=2, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    #
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/A级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/B级',
    #     label_A='食管炎-A级', label_B='食管炎-B级', y=1, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    #
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/A级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/D级',
    #     label_A='食管炎-A级', label_B='食管炎-D级', y=3, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    #
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/B级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/D级',
    #     label_A='食管炎-B级', label_B='食管炎-D级', y=3, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    #
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/B级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/C级',
    #     label_A='食管炎-B级', label_B='食管炎-C级', y=2, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    #
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/C级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/D级',
    #     label_A='食管炎-C级', label_B='食管炎-D级', y=3, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    #
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/D级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/A级',
    #     label_A='食管炎-D级', label_B='食管炎-A级', y=0, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    #
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/D级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/B级',
    #     label_A='食管炎-D级', label_B='食管炎-B级', y=1, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    #
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/D级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/C级',
    #     label_A='食管炎-D级', label_B='食管炎-C级', y=2, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    #
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/C级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/B级',
    #     label_A='食管炎-C级', label_B='食管炎-B级', y=1, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    #
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/C级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/A级',
    #     label_A='食管炎-C级', label_B='食管炎-A级', y=0, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)
    #
    # dataloader = create_dataset(
    #     data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/B级',
    #     data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_Cropped/A级',
    #     label_A='食管炎-B级', label_B='食管炎-A级', y=0, shuffle_AB=True, batchsize=batchsize,
    #     transform=transform, transform_A=transform, transform_B=transform)
    # dataloaders.append(dataloader)

    return dataloaders