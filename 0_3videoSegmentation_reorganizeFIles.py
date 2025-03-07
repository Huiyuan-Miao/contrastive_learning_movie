# organize the frame based on the correlation. adjacent frame representing the same scene is put in the same folder
import os
import shutil
import time
import numpy as np
import json
import scipy.io
import pandas as pd
import re

def cleanNum(a):
    temp = re.findall('scene[0-9.]*png',a)
    return temp[0]
count = 0
threshold = 0.5
for mov in ['E1780','E1782','E1783']:#['E1782','E1783']: #
    fromDir_ = '/mnt/HDD12TB/Miao/Datasets/theLifeofMammals/'+mov+'/VIDEO_TS/subSample30hz/snapshots'
    toDir_ = '/mnt/HDD12TB/Miao/Datasets/theLifeofMammals/subSample30hz_threshold'+str(threshold)+'/theLifeofMammals-'+mov+'-'
    dic=scipy.io.loadmat('/mnt/HDD12TB/Miao/Datasets/theLifeofMammals/'+mov+'/VIDEO_TS/subSample30hz/corrs.mat')
    isCopy = 1
    for i in range(0,len(dic['corr'][0])):
        nm = dic['path'][i][0].strip()
        c = dic['corr'][0][i]
        if count < 10:
            num = '000' + str(count)
        elif count<100:
            num = '00' + str(count)
        elif count <1000:
            num = '0' + str(count)
        elif count>=1000:
            num = str(count)
        toDir = toDir_ + num +'/'
        os.makedirs(toDir, exist_ok=True)
        if i == 0:
            shutil.copyfile(dic['path'][i][0].strip(), toDir+cleanNum(dic['path'][i][0].strip()))
        elif i == len(dic['corr'][0])-1:
            if dic['corr'][0][i]>= threshold:
                shutil.copyfile(dic['path'][i][0].strip(), toDir+cleanNum(dic['path'][i][0].strip()))
        else:
            if np.logical_and(dic['corr'][0][i]>= threshold,isCopy==1):
                shutil.copyfile(dic['path'][i][0].strip(), toDir+cleanNum(dic['path'][i][0].strip()))
            else:
                isCopy = 0
                if np.sum(dic['corr'][0][(i+1):(i+15)]> threshold) == 14:
                    count+=1
                    isCopy = 1
                    if count < 10:
                        num = '000' + str(count)
                    elif count < 100:
                        num = '00' + str(count)
                    elif count < 1000:
                        num = '0' + str(count)
                    elif count >= 1000:
                        num = str(count)
                    toDir = toDir_ + num + '/'
                    os.makedirs(toDir, exist_ok=True)
                    shutil.copyfile(dic['path'][i][0].strip(), toDir+cleanNum(dic['path'][i][0].strip()))



