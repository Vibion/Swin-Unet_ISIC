import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from datetime import datetime


def trainer_synapse(args, model, snapshot_path):
    # from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from datasets.ISIC_dataset import ISICDataset
    logging.basicConfig(filename=snapshot_path + f'log_{datetime.now().date()}.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # logging内容输出到控制台
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # transform
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor()
    ])
    # max_iterations = args.max_iterations
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    # 加载训练数据集
    db_train = ISICDataset(root=args.train_path,transform=transform)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)  # pin_memory固定内存
    
    # 加载验证数据集
    db_val = ISICDataset(root=args.val_path,transform=transform)
    print("The length of val set is: {}".format(len(db_val)))
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,worker_init_fn=worker_init_fn)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    ce_loss = CrossEntropyLoss() # 交叉熵
    dice_loss = DiceLoss(num_classes) # 
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    device = torch.device("cuda:0")
    iterator = tqdm(range(max_epoch), ncols=70) # 150
    # 开始训练
    for epoch_num in iterator:
        model.train()
        for i_batch, (img,mask,img_path) in enumerate(trainloader):
            image_batch, label_batch = img,mask
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch).to(device) # [batch_size,class,H,W]
            # 计算ce_loss时，outputs的形状为[batch*H*W,class],label_batch的形状为[batch*H*W]
            loss_ce = ce_loss(outputs.reshape(-1,2), label_batch.long().reshape(-1))
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(),loss_dice.item()))

            if iter_num % 20 == 0:
                print('开始保存tensorboard')
                image = image_batch[1, 0:3, :, :].squeeze()
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                # labs = label_batch[1, ...].unsqueeze(0) * 50
                labs = label_batch[1, ...] * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
                print('保存tensorboard完成')
                
                
        # # 开始验证
        model.eval()
        dice = 0
        logging.info('开始验证')
        for (img,mask,img_path) in valloader:
            image_batch, label_batch = img,mask
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            with torch.no_grad():
                outputs = model(image_batch).to(device)
            loss_ce = ce_loss(outputs.reshape(-1,2), label_batch.long().reshape(-1))
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss = 0.4 * loss_ce + 0.6 * loss_dice
            dice = 1- loss_dice + dice
            logging.info('dice : %f' % (1- loss_dice))
            # writer.add_scalar('info/val_loss', loss, iter_num)
            
        dice = dice / len(valloader)
        logging.info('epoch_num: %d : dice : %f' % (epoch_num, dice))
        if 'best_model.pth' not in os.listdir(snapshot_path):
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            # 保存最好的模型
            best_performance = dice
        if epoch_num % 5 == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        if dice > best_performance:
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            best_performance = dice
            logging.info(f'{epoch_num} epoch is the best model, dice is {best_performance}')
        logging.info('验证结束')
        

    writer.close()
    return "Training Finished!"