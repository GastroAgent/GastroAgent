import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/GasAgent-main')
from utils.data_utils import *
import torch
import argparse
import json
import torch
from conditional_flow_matcher import OptimalTransportConditionalFlowMatcher

class ImageGenerator:
    def __init__(self, args):
        self.args = args
        self.use_gt_vt = False
        self.args.solver = "euler"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net_model, vae, (text_model, text_tokenizer), (vision_model, process_single_image) = self._load_model(
            args.checkpoint, args.use_ema)
        self.net_model = net_model
        self.vae = vae
        self.text_model = text_model
        self.text_tokenizer = text_tokenizer
        self.vision_model = vision_model
        self.process_single_image = process_single_image
        self._setup_directories()
        self.intermediate_images = []  # Store intermediate images
        self.text_embeds_map = {}
        self.FM = OptimalTransportConditionalFlowMatcher(sigma=0, ot_method='exact')
        
    def _load_model(self, checkpoint: str, use_ema=False):
        """Initialize and load the model"""
        if True:
            net_model = None
            text_model = None
            text_tokenizer = None
            vision_model = None
            process_single_image = None
            vae = None
            return net_model, vae, (text_model, text_tokenizer), (vision_model, process_single_image)
 
    def generate(self):
        """Main generation loop"""
        eval_dataset = json.load(open(self.args.data_path, "r"))
        output_path = self.args.output_path
        k = 2
        dataset = []
        total = 0
        for data in eval_dataset:
            if not os.path.exists(data["x0"]):
                continue
            total += 1
            for i in range(len(data['x1_dirs'])):
                x1_paths = os.listdir(data['x1_dirs'][i])
                for x1_path in x1_paths[:k]:
                    x1_path = os.path.join(data['x1_dirs'][i], x1_path)
                    if os.path.exists(x1_path):
                        new_data = {}
                        new_data["question_id"] = data["question_id"]
                        new_data["x0"] = data["x0"]
                        new_data["label_A"] = data["label_A"]
                        new_data["label_A_id"] = data["label_A_id"]
                        new_data["label_B"] = data["x1_labels"][i]
                        new_data["label_B_id"] = data["label_B_ids"][i]
                        new_data["caption"] = data["caption"][i]
                        new_data["x1"] = x1_path
                        new_data["hint_path"] = x1_path
                        dataset.append(new_data)
                        
        with open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/images_dia_exam_flatten.json', 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        print(f'Saved to {output_path}')

def parse_args():
    parser = argparse.ArgumentParser(description='Sampling script for CFM model')
    parser.add_argument('--data_path', type=str,
                        default='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/images_dia_exam.json',
                        help='数据路径')
    parser.add_argument('--output_path', type=str,
                        default='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/images_dia_exam_flatten.json',
                        help='Directory to save generated images')
    return parser.parse_args()

def main():
    args = parse_args()
    generator = ImageGenerator(args)
    generator.generate()

if __name__ == "__main__":
    main()

"""
python generate_samples_grid.py --checkpoint outputs/results_otcfm_32_otcfm-large-batch_exp/otcfm/otcfm_weights_step_2000000.pt  --num_samples 4 --batch_size 4 --output_dir sample_ot-cfm_large_batch --image_size 128 128 --num_steps 8 --use_ema --solver heun --save_grid --save_intermediates --intermediate_freq 
""" 