import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config
from torchvision import transforms
from utils import DiceLoss
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str,
                    default='../data/', help='root dir for test data')  # for acdc volume_path=root_dir
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--output_dir',default='./model_out' ,type=str, help='output dir')   
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)


def inference(args, model):
    # 加载图片
    img = Image.open(args.img_path).convert('RGB')
    h,w = img.size
    # 定义transform
    transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor()
    ])
    # 进行transform和添加batch维度
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    # 定义device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    print("开始预测")
    with torch.no_grad():
        outputs = model(img.to(device)).to(device)
    # outputs = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
    outputs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
    # 如果类别1的概率大于0.7，则预测为类别1
    outputs = outputs[1]
    outputs[outputs>0.95] = 1
    outputs[outputs<=0.95] = 0
    # ndarray转换为PIL.Image
    outputs = np.uint8(outputs*255)
    # 按照h,w进行resize
    outputs = Image.fromarray(outputs).resize((w,h))
    outputs.save('./outputs.jpg')
    print("预测结果已保存在outputs.jpg")

if __name__ == "__main__":

    # 保证实验结果可以复现
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 实例化网络
    args.is_pretrain = True
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet",msg)

    # 加载单张图片
    img_path = '../data/test/images/ISIC_0012169.jpg'
    args.img_path = img_path
    
    inference(args, net)
