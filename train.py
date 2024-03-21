import os
import warnings
import numpy as np
import argparse
import math
import warnings
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils import data
import random
import cv2
import copy
from typing import Optional, List
import torch.nn.functional as F
from torch import nn, Tensor
from PIL import Image
import PIL.ImageOps
from tqdm import tqdm


# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=4,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
                    help='adam: decay of second order momentum of gradient')
parser.add_argument('--base_channels', type=int, default=128,
                    help='number of feature maps')
parser.add_argument('--seed', type=int, default=2017, help='random seed')
parser.add_argument('--ext_flag', action='store_true',
                    help='whether to use external factors')
parser.add_argument('--map_width', type=int, default=64,
                    help='image width')
parser.add_argument('--map_height', type=int, default=64,
                    help='image height')
parser.add_argument('--channels', type=int, default=2,
                    # (inflow + outflow) | XiAn & Chengdu: 2 channel | beijing:1 channel
                    help='number of flow image channels')
parser.add_argument('--dataset_name', type=str, default='ChengDu',
                    help='which dataset to use')
parser.add_argument('--city_road_map', type=str, default='cdu',
                    help='which city_road_map to use')
parser.add_argument('--run_num', type=int, default=0,
                    help='save model folder serial number')
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--point', default=64, type=int,
                    help="Number of bottleneck point")
parser.add_argument('--resnum', default=7, type=int,
                    help="Number of residual blocks")
parser.add_argument('--attnum', default=4, type=int,
                    help="Number of attention blocks")

args = parser.parse_known_args()[0]
args.ext_flag = True
print(args)

torch.manual_seed(args.seed)
warnings.filterwarnings('ignore')

# path for saving model---------------------------------------------
while os.path.exists('model/{}-{}-{}'.format(args.dataset_name,
                                             args.ext_flag,
                                             args.run_num)): args.run_num += 1
save_path = 'model/{}-{}-{}'.format(args.dataset_name,
                                    args.ext_flag,
                                    args.run_num)
print(save_path)
os.makedirs(save_path, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# # # initial model
model = MSSCL(args)

model.apply(weights_init_normal)
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
print_model_parm_nums(model, 'MSSCL')

criterion = nn.MSELoss()
criterion1 = nn.MSELoss()
if cuda:
    model.cuda()
    criterion.cuda()

# load training set and validation set----------------------
source_datapath = os.path.join('../data', args.dataset_name)
train_dataloader = get_dataloader_sr(
    source_datapath, args.batch_size, 'train', args.city_road_map, args.channels)
valid_dataloader = get_dataloader_sr(
    source_datapath, args.batch_size, 'valid', args.city_road_map, args.channels)
test_dataloader = get_dataloader_sr(
    source_datapath, args.batch_size, 'test', args.city_road_map, args.channels)

# Optimizers----------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# training phase ========================================
iter = 0
rmses = [np.inf]
maes = [np.inf]
mapes = [np.inf]
mapes_in, mapes_out = [np.inf], [np.inf]
last_out = []

# # adjust lr
dic = {30: 1e-4}
print('leaining rate changes when [epoch/lr]:', dic)

# start trainning --------------------------------------
pbar = tqdm(range(args.n_epochs))
for epoch in pbar:
    pbar.set_description(save_path[12:])

    # real_coarse_A:[4,2,64,64]   ext:[4,5]   real_fine_A:[4, 2, 64, 64]   road_A:[4, 1, 128, 128]
    for i, (real_coarse_A, ext, real_fine_A, road_A) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()
        gen_hr, loss1 = model(real_coarse_A, ext, road_A)  # [4, 2, 64, 64]
        loss = criterion(gen_hr, real_fine_A)
        (loss + loss1).backward()
        optimizer.step()

        iter += 1

        # validation----------------------------------------
        if iter % len(train_dataloader) == 0:
            model.eval()
            total_mape = 0
            total_mape_in, total_mape_out = 0.0, 0.0
            with torch.no_grad():
                for j, (flows_c, ext, flows_f, road) in enumerate(valid_dataloader):
                    preds, _ = model(flows_c, ext, road)
                    preds_ = preds.cpu().detach().numpy()
                    flows_f_ = flows_f.cpu().detach().numpy()
                    total_mape += get_MAPE(preds_, flows_f_) * len(flows_c)

            mape = total_mape / len(valid_dataloader.dataset)

            # select best MAPES to preserve model
            if mape < np.min(mapes):
                tqdm.write("epoch\t{}\titer\t{}\tMAPE\t{:.6f}".format(epoch, iter, mape))
                torch.save(model.state_dict(), '{}/final_model.pt'.format(save_path))
                f = open('{}/results.txt'.format(save_path), 'a')
                f.write("epoch\t{}\titer\t{}\tMAPE\t{:.6f}\n".format(epoch, iter, mape))
                f.close()
            mapes.append(mape)
        if epoch in dic:
            lr = dic[epoch]
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(args.b1, args.b2))