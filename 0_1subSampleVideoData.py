import os
import shutil
import time
import numpy as np
fromDir_ = '/mnt/HDD12TB/Miao/Datasets/theLifeofMammals/E1783/VIDEO_TS/snapshots'
toDir_ = '/mnt/HDD12TB/Miao/Datasets/theLifeofMammals/E1783/VIDEO_TS/subSample30hz/snapshots'
copyfile = 'scene'

i = 1
t = time.time()
for i in range(1,600000,2):
    if int(np.floor(i/20000))<10:
        toDir = toDir_ + '0'+str(int(np.floor(i/20000)))+'/'
    else:
        toDir = toDir_ + str(int(np.floor(i/20000)))+'/'

    fromDir = fromDir_ + str(int(np.floor(i/20000)))+'/'
    os.makedirs(toDir, exist_ok=True)
    if i < 10:
        num = '0000' + str(i)
    elif i<100:
        num = '000' + str(i)
    elif i <1000:
        num = '00' + str(i)
    elif i < 10000:
        num = '0' + str(i)
    elif i>=10000:
        num = str(i)
    nm = copyfile + num + '.png'
    if os.path.isfile(fromDir+nm):
        shutil.copyfile(fromDir+nm, toDir+nm)
        # os.remove(fromDir+nm)
    #     t = time.time()
    # if time.time() - t > 30:
    #     i = 600001
