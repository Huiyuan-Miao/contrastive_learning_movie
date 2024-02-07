import os
import sys
import math
import time
import numpy
import matplotlib.pyplot as plt
import torch.utils.data as data

from torchvision import datasets
from PIL import Image

term_width = 0
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg = None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    # if current < total-1:
    #     sys.stdout.write('\r')
    # else:
    #     sys.stdout.write('\n')

    sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder"""

    # Override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # This is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        # Make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path, index))
        return tuple_with_path

def default_loader(path):
    image = Image.open(path)
    if len(image.size) == 2:
        image = image.convert('RGB')
    return image

class ImageFilelist(data.Dataset):
    def __init__(self, filelist, transform=None, loader=default_loader):
        self.filelist = filelist
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.filelist[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.filelist)

def plot_training_stats(filename):
    file = open(filename, 'r')

    train = {'epoch':[], 'lr':[], 'loss':[], 'accuracy1':[], 'accuracy5':[]}
    val = {'epoch':[], 'lr':[], 'loss':[], 'accuracy1':[], 'accuracy5':[]}

    for line in file:
        line = line.replace('\n', '')
        data = line.split(' ')

        if 'Train' in line:
            train['epoch'].append(float(data[2][1:-1]))
            train['lr'].append(float(data[4][1:-1]))
            train['loss'].append(float(data[6][1:-1]))
            train['accuracy1'].append(float(data[8][1:-1]))
            train['accuracy5'].append(float(data[10][1:-1]))
        elif 'Validation' in line:
            val['epoch'].append(float(data[2][1:-1]))
            val['lr'].append(float(data[4][1:-1]))
            val['loss'].append(float(data[6][1:-1]))
            val['accuracy1'].append(float(data[8][1:-1]))
            val['accuracy5'].append(float(data[10][1:-1]))

    nrows, ncols = 1, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))

    #### LR
    axes[0].plot(train['epoch'], train['lr'])
    axes[0].plot(val['epoch'], val['lr'])
    axes[0].set_title('Learning rate')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(('Train', 'Validation'))

    #### Loss
    axes[1].plot(train['epoch'], train['loss'])
    axes[1].plot(val['epoch'], val['loss'])
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(('Train', 'Validation'))

    #### Top1 accuracy
    axes[2].plot(train['epoch'], train['accuracy1'])
    axes[2].plot(val['epoch'], val['accuracy1'])
    axes[2].set_title('Top1 accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim([0, 1])
    axes[2].yaxis.set_ticks(numpy.arange(0, 1, step=0.2))
    axes[2].legend(('Train', 'Validation'))

    #### Top5 accuracy
    axes[3].plot(train['epoch'], train['accuracy5'])
    axes[3].plot(val['epoch'], val['accuracy5'])
    axes[3].set_title('Top5 accuracy')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylim([0, 1])
    axes[3].yaxis.set_ticks(numpy.arange(0, 1, step=0.2))
    axes[3].legend(('Train', 'Validation'))

    plt.show()

if __name__ == '__main__':
    plot_training_stats('/media/dave/HDD12TB/Hojin/Project/PycharmProjects/Cadena/v1/AlexNet/training_stats.txt')