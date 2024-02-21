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

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/', help='root dir for test data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='ISIC', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', default='./model_out', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', default=False, help='whether to save results during inference')
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


def inference(args, model, test_save_path=None):
    from datasets.ISIC_dataset import ISICDataset
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor()
    ])
    db_test = ISICDataset(root=args.test_path,transform=transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dice_loss = DiceLoss(2)
    model.eval()
    metric = {'dice':0,'jaccard':0,'acc':0,'sens':0,'spec':0}
    
    for i_batch, (img,mask,img_path) in tqdm(enumerate(testloader)):
        image_batch, label_batch = img,mask
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        with torch.no_grad():
            outputs = model(image_batch).to(device)
        
        # DICE系数
        dice = 1 - dice_loss(outputs, label_batch, softmax=True)
        metric['dice'] += dice
        # sensitivity,specificity,accuracy
        pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
        gt = label_batch.squeeze(0).cpu().detach().numpy()
        # 生成蒙版接着按位与操作求的总和
        TP = ((pred == 1) & (gt == 1)).sum()
        TN = ((pred == 0) & (gt == 0)).sum()
        FP = ((pred == 1) & (gt == 0)).sum()
        FN = ((pred == 0) & (gt == 1)).sum()
        acc = (TP + TN) / (TP + TN + FP + FN)
        sens = TP / (TP + FN)
        if (TN + FP) != 0:
            spec = TN / (TN + FP)
        else:
            spec = 0
        metric['acc'] += acc
        metric['sens'] += sens
        metric['spec'] += spec
        
        # Jaccard系数
        intersection = np.logical_and(gt, pred)
        union = np.logical_or(gt, pred)
        jaccard = np.sum(intersection) / np.sum(union)
        metric['jaccard'] += jaccard
        # print("dice: ",dice)
    print("The average dice is: ",metric['dice']/len(testloader))
    print("The average acc is: ",metric['acc']/len(testloader))
    print("The average sens is: ",metric['sens']/len(testloader))
    print("The average spec is: ",metric['spec']/len(testloader))
    print("The average jaccard is: ",metric['jaccard']/len(testloader))
        
        
    # metric_list = metric_list / len(db_test)
    # for i in range(1, args.num_classes):
    #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    # performance = np.mean(metric_list, axis=0)[0]
    # mean_hd95 = np.mean(metric_list, axis=0)[1]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    # return "Testing Finished!"


if __name__ == "__main__":

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

    dataset_config = {
        'ISIC': {
            'root_path': args.root_path,
            # 'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'train_path': os.path.join(args.root_path,'train'),
            'val_path': os.path.join(args.root_path,'val'),
            'test_path': os.path.join(args.root_path,'test'),
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.test_path = dataset_config[dataset_name]['test_path']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


# ../data/test/images/ISIC_0021956.jpg