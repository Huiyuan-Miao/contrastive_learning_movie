''' Summary
To train AlexNet to fit Cadena's nerual data.
'''

import os
import sys
import random
import time
import numpy
import scipy.io
import matplotlib.pyplot as plt
import collections
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import json
from utils_2 import *
from alexnet_classificationOnly import *

# global image_fft_avg_mag # Average amplitude spectrum
# image_fft_avg = scipy.io.loadmat('./image_fft_avg.mat')
# image_fft_avg_mag = image_fft_avg['image_fft_avg_mag']
def main(whichModel):

    #### Parameters ####################################################################################################
    model_path = '/home/tonglab/Miao/pycharm/contrastiveLearning/AlexNet_supervisedLearning_imagenet_1000cate/'

    train_batch_size = 128
    val_batch_size = 128
    start_epoch = 0
    num_epochs = 50
    save_every_epoch = 10
    initial_learning_rate = 1e-2
    gpu_ids = [0]

    #### Create/Load model #############################################################################################
    # 1. If pre-trained models used without pre-trained weights. e.g., model = models.vgg19()
    # 2. If pre-trained models used with pre-trained weights. e.g., model = models.vgg19(pretrained=True)
    # 3. If our models used.
    ####################################################################################################################

    # model = models.alexnet(pretrained=False)
    model = AlexNet()


    if len(gpu_ids) > 1:
        # model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0]).cuda()
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
    elif len(gpu_ids) == 1:
        device = torch.device('cuda:%d'%(gpu_ids[0]))
        torch.cuda.set_device(device)
        model.cuda()
        model.to(device)

    loss_function = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=1e-4)

    #### Resume from checkpoint
    try:
        os.mkdir(model_path)
    except:
        pass

    load_path = os.path.join(model_path,whichModel)
    if os.path.isfile(load_path):
        print("... Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        start_epoch = checkpoint['epoch'] + 1

        if len(gpu_ids) <= 1: # 1. Multi-GPU to single-GPU or -CPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
            checkpoint['state_dict'] = new_state_dict
        else: # 2. single-GPU or -CPU to Multi-GPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'module.' in k:
                    name = k
                else:
                    name = 'module.' + k
                new_state_dict[name] = v
            checkpoint['state_dict'] = new_state_dict

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param in optimizer.param_groups:
            param['initial_lr'] = initial_learning_rate
    else:
        print("... No checkpoint found at '{}'".format(load_path))
        print("train from start")

        # with torch.no_grad():
        #     model.features[0].weight.copy_(checkpoint['state_dict']['features.0.weight'])
        #     model.features[0].bias.copy_(checkpoint['state_dict']['features.0.bias'])
        # model.features[0].weight.requires_grad = False # this is for Alexnet_v3 freeze layer 1 weight as the original alexnet weight
        # model.features[0].bias.requires_grad = False # this is for Alexnet_v3

    #### Learning rate scheduler
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[61], gamma=0.1, last_epoch=start_epoch-1)
    # lr_scheduler.step()

    #### Data loader ###################################################################################################
    # 0, n01622779, Owl; 1, n02123045, Cat; 2, n02129165, Lion; 3, n02132136, Bear; 4, n02326432, Hare;
    # 5, n02342885, Hamster; 6, n02410509, Bison; 7, n02504458, Elephant; 8, n02690373, Airliner;
    # 9, n03594945, Jeep; 10, n04147183, Schooner; 11, n04273569, Speedboat; 12, n04285008, Sports car;
    # 13, n04344873, Couch; 14, n04380533, Table lamp, 15, n04398044, Teapot
    ####################################################################################################################

    # train_dataset = torchvision.datasets.ImageFolder(
    #     # "/home/tonglab/Documents/Data/ILSVRC2012/images/train_16",
    #     "/home/tonglab/Datasets/imagenet"+str(1000)+'/train',
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         # transforms.Grayscale(),
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.449], std=[0.226])
    #         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ]))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # val_dataset = torchvision.datasets.ImageFolder(
    val_dataset = ImageFolderWithPaths(
        # "/home/tonglab/Documents/Data/ILSVRC2012/images/val_16",
        '/mnt/HDD12TB/Miao/Datasets/theLifeofMammals/E1783/VIDEO_TS/subSample30hz/',
        # "/home/tonglab/Datasets/imagenet" + str(1000) + '/val',
        transforms.Compose([
            transforms.CenterCrop((224)),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.449], std=[0.226])

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    #### Train/Val #####################################################################################################
    c,p = val(val_loader, model, loss_function, optimizer, gpu_ids)
    dic = {'corr':np.array(c),'path':p}
    scipy.io.savemat('/mnt/HDD12TB/Miao/Datasets/theLifeofMammals/E1783/VIDEO_TS/subSample30hz/corrs.mat',dic)

def val(val_loader, model, loss_function, optimizer, gpu_ids):
    c = []
    p = []
    model.eval()
    for batch_index, (inputs, targets, paths, indices) in enumerate(val_loader):
        if np.mod(batch_index,1000) == 0:
            print(batch_index)
            dic = {'corr': np.array(c), 'path': p}
            scipy.io.savemat('/mnt/HDD12TB/Miao/Datasets/theLifeofMammals/E1783/VIDEO_TS/subSample30hz/corrs.mat',dic)
        if len(gpu_ids) >= 1:
            inputs = inputs.cuda(non_blocking=True)
        outputs = model(inputs)
        if batch_index == 0:
            last_outputs = outputs
            c.append(0)
        else:
            # print(scipy.stats.pearsonr(outputs[0].detach().cpu(), last_outputs[0].detach().cpu()))
            c.append(scipy.stats.pearsonr(outputs[0].detach().cpu(), last_outputs[0].detach().cpu())[0])
        p.append(paths)
    return c,p


def is_correct(output, target, topk=1):
    with torch.no_grad():
        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        num_correct = correct[:topk].contiguous().view(-1).float().sum(0, keepdim=True) # top5 glitch

        return correct, num_correct


if __name__ == '__main__':
    # for i in ['ckpt_latest_trainForClassification.pth','ckpt_latest_finetuneWithDualErr.pth']:
    for i in ['checkpoint.pth.tar']:
        main(whichModel = i)