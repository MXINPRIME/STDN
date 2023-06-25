import os
import glob
import random

dir = 'E:/Zhangziye/USOD10K/RGB/'

def get_list():
    ftest = open('list/USOD10K.txt', 'w')
    trainlist = glob.glob(dir+'*.png')
    print(len(trainlist))
    random.shuffle(trainlist)

    for idx in trainlist:
        ftest.write(os.path.basename(str(idx)) +'\n')

get_list()


# def get_list():
#     ftest = open('./urpc_all.txt', 'w')
#     trainlist = glob.glob(dir+'*.jpg')
#     # random.shuffle(trainlist)
#     # trainlist = sorted(trainlist, key=lambda s: int(os.path.basename(s).split('_')[0].split('.')[0]))
#     trainlist = sorted(trainlist)
#     for idx in trainlist:
#         ftest.write(os.path.basename(str(idx)) +'\n')
#     print(len(trainlist))
# get_list()