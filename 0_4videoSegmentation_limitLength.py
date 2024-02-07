import os
import shutil
import time
import numpy as np
import json
import scipy.io
import pandas as pd
import re
threshold = 0.5
freq = 10
length = 0.3 # sec
fromDir_ = '/media/tonglab/789aa685-f3c0-4b78-afb5-88d669eb7059/Miao/Data/theLifeOfMammals/subSample'+str(freq)+'hz_threshold'+str(threshold)+'/'
toDir_ = '/media/tonglab/789aa685-f3c0-4b78-afb5-88d669eb7059/Miao/Data/theLifeOfMammals/subSample'+str(freq)+'hz_threshold'+str(threshold)+'_length'+str(length)+'/'
# f = open('/media/tonglab/789aa685-f3c0-4b78-afb5-88d669eb7059/Miao/Data/E1783/VIDEO_TS/subSample10hz/corrs.json', "w")
# dic = json.load(f)
def cleanNum(a):
    temp = re.findall('scene[0-9.]*png',a)
    return temp[0]
count = 0
fs = os.listdir(fromDir_)
fs = np.sort(fs)
for f in fs:
    imgs = os.listdir(fromDir_+f)
    fromDir = fromDir_+f+'/'
    l = len(imgs)
    imgs_sort = np.sort(imgs)
    count = 0
    count2 = 0
    start = 0
    toDir = toDir_ + f + '-' + str(int(np.round(count / 2))) + '/'
    os.makedirs(toDir, exist_ok=True)
    for i in range(l):
        if count2 == freq * length:
            count2 = 0
            count+=1
            toDir = toDir_ + f + '-' + str(int(np.round(count / 2))) + '/'
            start = i
        if np.logical_and(np.mod(count,2) == 0,l-start>=5):
            os.makedirs(toDir, exist_ok=True)
            shutil.copyfile(fromDir+imgs_sort[i], toDir + imgs_sort[i])
        count2 +=1

