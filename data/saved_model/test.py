import os
import sys
import warnings
import numpy as np
import argparse
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from model import Supreme, weights_init_normal
from utils import get_MSE, get_MAE, get_MAPE, mask_loss, print_model_parm_nums


# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32,
                    help='training batch size')
parser.add_argument('--n_residuals', type=int, default=16,
                    help='number of residual units')
parser.add_argument('--base_channels', type=int,
                    default=64, help='number of feature maps')
parser.add_argument('--img_width', type=int, default=10,
                    help='image width')
parser.add_argument('--img_height', type=int, default=20,
                    help='image height')
parser.add_argument('--depth', type=int, default=6,
                    help='number of historical radiomaps')
parser.add_argument('--channels', type=int, default=1,
                    help='number of radiomap channels')
parser.add_argument('--zoom', type=int,
                    default=2, help='upscale factor')
parser.add_argument('--scaler_X', type=int, default=-120,
                    help='scaler of coarse-grained radiomaps')
parser.add_argument('--scaler_Y', type=int, default=-120,
                    help='scaler of fine-grained radiomaps')
parser.add_argument('--ext_dim', type=int, default=7,
                    help='external factor dimension')
parser.add_argument('--ext_flag', action='store_true',
                    help='whether to use external factors')
parser.add_argument('--has_CLH',type=bool,default = True,
                    help='whether to use cross connection')
parser.add_argument('--dataset_test', type=str, default='zjg/test',
                    help='test dataset to use')
parser.add_argument('--mask_loss_flag', type=bool,default=True,
                    help='whether to use masked loss')

opt = parser.parse_args()
print(opt)
warnings.filterwarnings('ignore')

if opt.ext_flag == True:
    model_path = 'data/saved_model/zjg_10_20_ext'+'_'+str(opt.n_residuals)+'_'+str(opt.base_channels)+'_'+str(opt.depth)+'.pt'
else:
    model_path = 'data/saved_model/zjg_10_20'+'_'+str(opt.n_residuals)+'_'+str(opt.base_channels)+'_'+str(opt.depth)+'.pt'

# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# load model
model = Supreme(in_channels=opt.channels,
                out_channels=opt.channels,
                img_width=opt.img_width,
                img_height=opt.img_height,
                depth = opt.depth,
                zoom = opt.zoom,
                n_residual_blocks=opt.n_residuals,
                base_channels=opt.base_channels,
                ext_flag=opt.ext_flag, 
               )
model.load_state_dict(torch.load(model_path))
model.eval()
if cuda:
    model.cuda()
print_model_parm_nums(model, 'Supreme')

#load test set
datapath = os.path.join('data', opt.dataset_test)
X_test = np.load(os.path.join(datapath, 'X10.npy'))/opt.scaler_X
Y_test = np.load(os.path.join(datapath, 'X20.npy'))
mask_test = np.load(os.path.join(datapath, 'mask20.npy'))
ext_test =  np.load(os.path.join(datapath, 'ext.npy'))
test_size = len(X_test)

X_t = np.zeros((test_size-opt.depth+1,opt.depth,X_test.shape[1],X_test.shape[2]))
Y_t= np.zeros((test_size-opt.depth+1,1,Y_test.shape[1],Y_test.shape[2]))
ext_t = np.zeros((test_size-opt.depth+1,ext_test.shape[1]))
mask_t = np.zeros((test_size-opt.depth+1,1,Y_test.shape[1],Y_test.shape[2]))
for k in range(test_size-opt.depth+1):
    X_t[k,:,:,:] = X_test[k:k+opt.depth]
    Y_t[k,:,:] = Y_test[(k+opt.depth-1)]
    mask_t[k,:,:] = mask_test[(k+opt.depth-1)]
    ext_t[k,:] = ext_test[(k+opt.depth-1)]
X_t = np.expand_dims(X_t,1)


total_mse, total_mae, total_mape = 0, 0, 0
l=0
while l<test_size - opt.depth+1:
    X_tbatch = Tensor(X_t[l:l+opt.batch_size])
    Y_tbatch = Tensor(Y_t[l:l+opt.batch_size])
    ext_tbatch = Tensor(ext_t[l:l+opt.batch_size])
    mask_tbatch = mask_t[l:l+opt.batch_size]
    preds = model(X_tbatch,ext_tbatch)
    preds = preds.cpu().detach().numpy()* opt.scaler_Y
    rm_f = Y_tbatch.cpu().detach().numpy()
    total_mse += get_MSE(preds, rm_f,mask_tbatch)
    total_mae += get_MAE(preds, rm_f,mask_tbatch)
    total_mape += get_MAPE(preds, rm_f,mask_tbatch)
    l=l+opt.batch_size
    
rmse = np.sqrt(total_mse/np.sum(mask_t))
mae = total_mae/np.sum(mask_t)
mape = total_mape/np.sum(mask_t)

print('Test RMSE = {:.6f}, MAE = {:.6f}, MAPE = {:.6f}'.format(rmse, mae, mape))

