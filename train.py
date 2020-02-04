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
parser.add_argument('--n_epochs', type=int, default=50,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
                    help='adam: decay of second order momentum of gradient')
parser.add_argument('--n_residuals', type=int, default=16,
                    help='number of residual units')
parser.add_argument('--base_channels', type=int,
                    default=64, help='number of feature maps')
parser.add_argument('--img_width', type=int, default=10,
                    help='radio map width')
parser.add_argument('--img_height', type=int, default=20,
                    help='radio map height')
parser.add_argument('--depth', type=int, default=6,
                    help='number of historical radio maps')
parser.add_argument('--channels', type=int, default=1,
                    help='number of radio map channels')
parser.add_argument('--sample_interval', type=int, default=20,
                    help='interval between validation')
parser.add_argument('--harved_epoch', type=int, default=30,
                    help='halved at every x interval')
parser.add_argument('--zoom', type=int,
                    default=2, help='upscale factor')
parser.add_argument('--seed', type=int, default=2017, help='random seed')
parser.add_argument('--scaler_X', type=int, default=-120,
                    help='scaler of coarse-grained radiomaps')
parser.add_argument('--scaler_Y', type=int, default=-120,
                    help='scaler of fine-grained radiomaps')
parser.add_argument('--ext_dim', type=int, default=7,
                    help='external factor dimension')
parser.add_argument('--ext_flag',  action='store_true',
                    help='whether to use external factors')
parser.add_argument('--dataset_t', type=str, default='zjg/train',
                    help='training dataset to use')
parser.add_argument('--dataset_v', type=str, default='zjg/valid',
                    help='valid dataset to use')
parser.add_argument('--save_model', type=bool,default=True,
                    help='whether to use save model')


opt = parser.parse_args()
print(opt)
torch.manual_seed(opt.seed)

# path for saving model 
if opt.ext_flag == True:
    save_path = 'data/saved_model/zjg_10_20_ext'+'_'+str(opt.n_residuals)+'_'+str(opt.base_channels)+'_'+str(opt.depth)
else:
    save_path = 'data/saved_model/zjg_10_20'+'_'+str(opt.n_residuals)+'_'+str(opt.base_channels)+'_'+str(opt.depth)
#os.makedirs(save_path, exist_ok=True)

# test CUDA available
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# initial proposed model
model = Supreme(in_channels=opt.channels,
                out_channels=opt.channels,
                img_width=opt.img_width,
                img_height=opt.img_height,
                n_residual_blocks=opt.n_residuals,
                base_channels=opt.base_channels,
                zoom=opt.zoom,
                depth=opt.depth,
                ext_flag=opt.ext_flag)
model.apply(weights_init_normal)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
print_model_parm_nums(model, 'Supreme')

if cuda:
    model.cuda()
    
# load training and validation set
datapath = os.path.join('data', opt.dataset_v)
X_valid = np.load(os.path.join(datapath, 'X10.npy'))/opt.scaler_X
Y_valid = np.load(os.path.join(datapath, 'X20.npy'))/opt.scaler_Y
mask_valid = np.load(os.path.join(datapath, 'mask20.npy'))
ext_valid =  np.load(os.path.join(datapath, 'ext.npy'))
valid_size = len(X_valid)

datapath = os.path.join('data', opt.dataset_t)
X_train = np.load(os.path.join(datapath, 'X10.npy'))/opt.scaler_X
Y_train = np.load(os.path.join(datapath, 'X20.npy'))/opt.scaler_Y
mask_train = np.load(os.path.join(datapath, 'mask20.npy'))
ext_train =  np.load(os.path.join(datapath, 'ext.npy'))
train_size = len(X_train)

X_v = np.zeros((valid_size-opt.depth+1,opt.depth,X_valid.shape[1],X_valid.shape[2]))
Y_v = np.zeros((valid_size-opt.depth+1,1,Y_valid.shape[1],Y_valid.shape[2]))
ext_v = np.zeros((valid_size-opt.depth+1,ext_valid.shape[1]))
mask_v = np.zeros((valid_size-opt.depth+1,1,Y_valid.shape[1],Y_valid.shape[2]))
for k in range(valid_size-opt.depth+1):
    X_v[k,:,:,:] = X_valid[k:k+opt.depth]
    Y_v[k,:,:] = Y_valid[(k+opt.depth-1)]
    mask_v[k,:,:] = mask_valid[(k+opt.depth-1)]
    ext_v[k,:] = ext_valid[(k+opt.depth-1)]
X_v = np.expand_dims(X_v,1)



# Optimizers
optimizer = torch.optim.Adam(
    model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

#training phase
iter_per_epoch = int(np.ceil(train_size * 1. / opt.batch_size)) 
n_iter = 0
rmses = [np.inf]
maes = [np.inf]
mapes = [np.inf]
for epoch in range(opt.n_epochs):
    perm_idx = np.random.permutation(train_size - opt.depth+1) 
    j =0
    while j<train_size - opt.depth+1:
        batch_idx = perm_idx[j:(j + opt.batch_size)]
        X_batch = np.zeros((len(batch_idx),opt.depth, X_train.shape[1],X_train.shape[2]))
        Y_batch = np.zeros((len(batch_idx),1, Y_train.shape[1],Y_train.shape[2]))
        mask_batch =  np.zeros((len(batch_idx),1, Y_train.shape[1],Y_train.shape[2]))
        ext_batch = np.zeros((len(batch_idx),ext_train.shape[1]))
        for k in range(len(batch_idx)):
            X_batch[k, :, : , :] = X_train[batch_idx[k] : (batch_idx[k] + opt.depth )]
            Y_batch[k, :,:,:] = Y_train[(batch_idx[k]+opt.depth-1)]
            mask_batch[k, :,:,:] = mask_train[(batch_idx[k]+opt.depth-1)]
            ext_batch[k,:] = ext_train[(batch_idx[k]+opt.depth-1)]
        X_batch = Tensor(np.expand_dims(X_batch,1))
        Y_batch = Tensor(Y_batch)
        mask_batch = Tensor(mask_batch)
        ext_batch = Tensor(ext_batch)
        model.train()
        optimizer.zero_grad()
        # generate images with high resolution
        gen_hr = model(X_batch,ext_batch)
        loss = mask_loss(gen_hr, Y_batch,mask_batch)
        loss.backward()
        optimizer.step()

        print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch,
                                                     opt.n_epochs,
                                                     j/opt.batch_size,
                                                     iter_per_epoch,    
                                                     np.sqrt(loss.item())))
        j += opt.batch_size
        
        n_iter+=1
        
        # validation phase
        if n_iter % opt.sample_interval == 0:
            model.eval()
            valid_time = datetime.now()
            total_mse, total_mae, total_mape = 0, 0, 0
            l = 0
            while l<valid_size - opt.depth+1:
                X_vbatch = Tensor(X_v[l:l+opt.batch_size])
                Y_vbatch = Tensor(Y_v[l:l+opt.batch_size])
                ext_vbatch = Tensor(ext_v[l:l+opt.batch_size])
                mask_vbatch = mask_v[l:l+opt.batch_size]
                preds = model(X_vbatch,ext_vbatch)
                preds = preds.cpu().detach().numpy()* opt.scaler_Y

                flows_f = Y_vbatch.cpu().detach().numpy()* opt.scaler_Y
                total_mse += get_MSE(preds, flows_f,mask_vbatch)
                total_mae += get_MAE(preds, flows_f,mask_vbatch)
                total_mape += get_MAPE(preds, flows_f,mask_vbatch)
                l=l+opt.batch_size
            rmse = np.sqrt(total_mse/np.sum(mask_v))
            mae = total_mae/np.sum(mask_v)
            mape = total_mape/np.sum(mask_v)
            if rmse < np.min(rmses):
                print("iter\t{}\tRMSE\t{:.6f}\ttime\t{}".format(n_iter, rmse, datetime.now()-valid_time))
                print("iter\t{}\tMAE\t{:.6f}\ttime\t{}".format(n_iter, mae, datetime.now()-valid_time))
                print("iter\t{}\tMAPE\t{:.6f}\ttime\t{}".format(n_iter, mape, datetime.now()-valid_time))
                # save model 
                if opt.save_model:
                    torch.save(model.state_dict(),save_path+'.pt')
            #save the validation history         
            rmses.append(rmse)
            maes.append(mae)
            mapes.append(mape)
    # periodically decrease the learning rate
    if epoch % opt.harved_epoch == 0 and epoch != 0:
        opt.lr /= 2
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
  


