# test whether the model trained with contrastive learning learn a good representation from movie data - finetune the entire model to do imagenet classification

from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter, is_correct, accuracy
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from alexnet_dualErr import AlexNet
from vgg_dualErr import *
from losses import SupConLoss
from SupContrast_FromHojin.networks.resnet_big import SupConResNet
import collections

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='70,80,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='alexnet')
    parser.add_argument('--dataset', type=str, default='imageNet',
                        choices=['cifar10', 'cifar100', 'path','imageNet'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='/home/tonglab/Datasets/imagenet1000/train/', help='path to custom dataset')
    parser.add_argument('--val_data_folder', type=str, default='/home/tonglab/Datasets/imagenet1000/val/', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './SimCLR/{}_models'.format(opt.dataset)
    opt.tb_path = './SimCLR/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    elif opt.dataset == 'imageNet':
        mean = (0.485,0.456,0.406)
        std  = (0.229,0.224,0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4409], std=[0.226])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    elif opt.dataset == 'imageNet':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
        val_dataset = datasets.ImageFolder(root=opt.val_data_folder,
                                            transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    return train_loader,val_loader


def set_model(opt,gpu_ids):
    model = AlexNet()
    # model = SupConResNet(name=opt.model)
    criterion_ctr = SupConLoss(temperature=opt.temp)
    criterion_clf = nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model,device_ids=gpu_ids).cuda()
        elif len(gpu_ids) == 1:
            device = torch.device('cuda:%d'%(gpu_ids[0]))
            torch.cuda.set_device(device)
            model = model.cuda()
            model.to(device)
        criterion_ctr = criterion_ctr.cuda()
        criterion_clf = criterion_clf.cuda()
        cudnn.benchmark = True

    return model, criterion_ctr,criterion_clf


def train(train_loader, model, criterion_ctr,criterion_clf, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ctr = AverageMeter()
    losses_clf = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features,out = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss_ctr = criterion_ctr(features, labels)
        elif opt.method == 'SimCLR':
            loss_ctr = criterion_ctr(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))
        loss_clf = criterion_clf(out,labels.repeat(2))
        loss = loss_clf
        # update metric
        losses.update(loss.item(), bsz)
        losses_ctr.update(loss_ctr.item(), bsz)
        losses_clf.update(loss_clf.item(), bsz)
        topx_accuracy = accuracy(out, labels.repeat(2), topk=[1,5])
        top1.update(topx_accuracy[0][0],bsz*2)
        # top5_accuracy = accuracy(out, labels.repeat(2), topk=[5])
        top5.update(topx_accuracy[1][0], bsz*2)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lossTotal {loss.val:.3f} ({loss.avg:.3f})\t'
                  'lossContrast {loss_ctr.val:.3f} ({loss_ctr.avg:.3f})\t'
                  'lossClassification {loss_clf.val:.3f} ({loss_clf.avg:.3f})\t'
                  'top1_accuracy {top1.val:.6f} ({top1.avg:.6f})\t'
                  'top5_accuracy {top5.val:.6f} ({top5.avg:.6f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,loss_ctr = losses_ctr,loss_clf=losses_clf,
                   top1 = top1, top5 = top5))
            sys.stdout.flush()

    return losses_ctr.avg,losses_clf.avg,losses.avg,top1.avg,top5.avg

def val(val_loader, model, criterion_ctr,criterion_clf, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_clf = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(val_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        features,out = model(images)

        loss_clf = criterion_clf(out,labels)

        # update metric
        losses_clf.update(loss_clf.item(), bsz)
        topx_accuracy = accuracy(out, labels, topk=[1,5])
        top1.update(topx_accuracy[0][0],bsz)
        top5.update(topx_accuracy[1][0], bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Validation: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lossClassification {loss_clf.val:.3f} ({loss_clf.avg:.3f})\t'
                  'top1_accuracy {top1.val:.6f} ({top1.avg:.6f})\t'
                  'top5_accuracy {top5.val:.6f} ({top5.avg:.6f})'.format(
                   epoch, idx + 1, len(val_loader), batch_time=batch_time,
                   data_time=data_time,loss_clf=losses_clf,
                   top1 = top1, top5 = top5))
            sys.stdout.flush()

    return losses_clf.avg,top1.avg,top5.avg

def main():
    opt = parse_option()
    gpu_ids = [0]
    # build data loader
    train_loader,val_loader = set_loader(opt)
    start_epoch = 1
    # build model and criterion
    model, criterion_ctr,criterion_clf = set_model(opt,gpu_ids)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    load_path = os.path.join(opt.save_folder, 'ckpt_latest_finetuneWithClassification.pth')
    tuneBefore = 1
    if os.path.isfile(load_path) == False:
        load_path = os.path.join(opt.save_folder, 'ckpt_latest.pth')
        tuneBefore = 0
    if os.path.isfile(load_path):
        print("... Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        if tuneBefore == 1:
            start_epoch = checkpoint['epoch'] + 1

        if len(gpu_ids) <= 1: # 1. Multi-GPU to single-GPU or -CPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
            checkpoint['model'] = new_state_dict
        else: # 2. single-GPU or -CPU to Multi-GPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['model'].items():
                if 'module.' in k:
                    name = k
                else:
                    name = 'module.' + k
                new_state_dict[name] = v
            checkpoint['model'] = new_state_dict

        model.load_state_dict(checkpoint['model'])
        if tuneBefore == 1:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param in optimizer.param_groups:
                param['initial_lr'] = opt.learning_rate
        # # turn off some parameters
        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.classifier.parameters():
        #     param.requires_grad = True
    else:
        print("... No checkpoint found at '{}'".format(load_path))
        print("train from start")

    # training routine
    if tuneBefore == 0:
        time1_val = time.time()
        loss_clf_val, top1_val, top5_val = val(val_loader, model, criterion_ctr, criterion_clf, optimizer, 0, opt)
        time2_val = time.time()
        stat_file = open(os.path.join(opt.save_folder, 'training_stats_trainForClassification.txt'), 'a+')
        stat_str = '[Validation] Epoch ({}) LR ({:.4f}) LossClassificationd ({:.4f}) Top1Acc ({:.6f}) Top5Acc ({:.6f}) Time ({:.2f})'. \
            format(0, optimizer.param_groups[0]['lr'], loss_clf_val, top1_val, top5_val, time2_val - time1_val)
        stat_file.write(stat_str + '\n')
        stat_file.close()

    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss_ctr,loss_clf,loss,top1,top5 = train(train_loader, model, criterion_ctr,criterion_clf, optimizer, epoch, opt)
        time2 = time.time()
        time1_val = time.time()
        loss_clf_val,top1_val,top5_val = val(val_loader, model, criterion_ctr,criterion_clf, optimizer, epoch, opt)
        time2_val = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        stat_file = open(os.path.join(opt.save_folder, 'training_stats_finetuneWithClassification.txt'), 'a+')
        stat_str = '[Train] Epoch ({}) LR ({:.4f}) LossTotal ({:.4f}) LossContrast ({:.4f}) LossClassification ({:.4f}) Top1Acc ({:.6f}) Top5Acc ({:.6f}) Time ({:.2f})'. \
            format(epoch, optimizer.param_groups[0]['lr'], loss, loss_ctr, loss_clf, top1,top5, time2 - time1)
        stat_file.write(stat_str + '\n')
        stat_str = '[Validation] Epoch ({}) LR ({:.4f}) LossClassification ({:.4f}) Top1Acc ({:.6f}) Top5Acc ({:.6f}) Time ({:.2f})'. \
            format(epoch, optimizer.param_groups[0]['lr'], loss_clf_val,top1_val,top5_val, time2_val - time1_val)
        stat_file.write(stat_str + '\n')
        stat_file.close()

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}_finetuneWithClassification.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        save_file = os.path.join(
            opt.save_folder, 'ckpt_latest_finetuneWithClassification.pth')
        save_model(model, optimizer, opt, epoch, save_file)
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last_finetuneWithClassification.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
