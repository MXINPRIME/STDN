import argparse
import os
import numpy as np
import cv2
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
import torchvision
from torch.autograd import Variable
from data.underwater_dataset import DerainTrainData
from model.my_vgg import VGGFeature
from model.RCP_RGP_AAB_PYC  import UnderNet
from pytorch_msssim import  MS_SSIM
import sys
import logging
import random

import time
import utils
import glob


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--iter", type=int, default=100000)
parser.add_argument("--start_iter", type=int, default=0)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--dim_z", type=int, default=128)
parser.add_argument("--dim_class", type=int, default=128)
parser.add_argument("--rec_weight", type=float, default=0.2)
parser.add_argument("--div_weight", type=float, default=0.1)
parser.add_argument("--crop_prob", type=float, default=0.3)
parser.add_argument("--n_class", type=int, default=10)
parser.add_argument("--train_root", type=str, default=r'E:\Zhangziye\UIEB640\UIEB640')
parser.add_argument("--train_list", type=str, default='./data/list/UIEB_train.txt')
parser.add_argument("--patch_size", type=int, default=224)
parser.add_argument('--gpu', type=str, default='0', help='gpu_id')
parser.add_argument('--resume', type=str, default='', help='gpu_id')
args = parser.parse_args()
args.distributed = False

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.resume:
    save = args.resume
    checkpoint_path = save
else:
    save = '{}-{}'.format('underwater', time.strftime("%Y%m%d-%H%M%S"))
    checkpoint_path = './checkpoint/' + save
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
utils.create_exp_dir(checkpoint_path, scripts_to_save=glob.glob('*.py') + glob.glob('./model/*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(checkpoint_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info('Run file:  %s', __file__)
logging.info('args= %s', args)



def requires_grad(module, flag):
    for m in module.parameters():
        m.requires_grad = flag



def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)





def train(args, dataset_pair, dataset_unpair, gen, dis_l=None, dis_g=None):

    vgg = VGGFeature(True)
    vgg.load_state_dict(torch.load('./pretrain/vggfeature.pth'))
    vgg = vgg.cuda().eval()
    requires_grad(vgg, False)# TODO: do not update the parameter of the VGG16

    g_optim = optim.Adam(gen.parameters(), lr=5e-4, betas=(0, 0.999))

    loader = data.DataLoader(
        dataset_pair,
        batch_size=args.batch,
        num_workers=3,
        shuffle=True,
        drop_last=True,
    )
    loader_iter = sample_data(loader)
    gen.train()

    L1_loss = nn.L1Loss()
    L1_loss = L1_loss.cuda()

    ssim_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True)
    ssim_loss = ssim_loss.cuda()

    pbar = range(args.start_iter, args.iter)
    eps = 1e-8

    for i in pbar:
        input_img, real = next(loader_iter)

        real      = real.cuda()
        input_img = input_img.cuda()

        features = vgg(input_img)
        g_optim.zero_grad()
        fake1, _, _   = gen(input_img, features)
        mseloss = L1_loss(fake1, real)
        _ssim_loss = 1 - ssim_loss(real, fake1)

        g_loss = args.rec_weight * mseloss + _ssim_loss * 0.8
        g_loss.backward()
        g_optim.step()

        if ((i+1) % 50 ==0):
            logging.info('step: %d   r_loss: %.4f  mse: %.4f', i, _ssim_loss.item(), mseloss.item())

        if ((i+1) % 500 ==0):
            torchvision.utils.save_image(
                fake1,
                f"sample/{str(i).zfill(6)}.png",
                nrow=int(args.batch ** 0.5),
                normalize=True,
                range=(0, 1),
            )
            torchvision.utils.save_image(
                input_img,
                f"sample/{str(i).zfill(6)}_input.png",
                nrow=int(args.batch ** 0.5),
                normalize=True,
                range=(0, 1),
            )
            torchvision.utils.save_image(
                real,
                f"sample/{str(i).zfill(6)}_real.png",
                nrow=int(args.batch ** 0.5),
                normalize=True,
                range=(0, 1),
            )

        if ((i+1) % 10000 == 0):
            torch.save(
                {
                    "args": args,
                    "g": gen.state_dict(),
                },
                "{}/{:06d}.pt".format(checkpoint_path, i),
            )
        if ((i+1)% 8000==0):
            current_lr = g_optim.param_groups[0]['lr']
            for param_group in g_optim.param_groups:
                param_group['lr'] = current_lr / 5


if __name__ == "__main__":

    dataset_pair = DerainTrainData(args.train_root, args.train_list, crop_size=(args.patch_size, args.patch_size))
    gen = UnderNet(3, 3, ngf=32, weight=0.5).cuda()
    train(args, dataset_pair, None,  gen, None, None)
