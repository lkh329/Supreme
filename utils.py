import torch.nn as nn
import torch
import math
import numpy as np


def get_MSE(pred, real,mask):
    return np.sum(np.power(real - pred, 2)*mask)

def get_MAE(pred, real,mask):
    return np.sum(np.abs(real - pred)*mask)
    
def get_MAPE(pred, real,mask, upscale_factor=4):
    ori_real = real.copy()
    epsilon = 1 # if use small number like 1e-6 would result in very large value
    real[real == 0] = epsilon 
    return np.sum(np.abs((ori_real - pred) / real)*mask)

def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))
    
def mask_loss(output,target,mask):
    num = torch.sum(mask)
    mask =mask.double()
    initial_loss = nn.MSELoss(reduction='none')
    tempMSE = initial_loss(output,target).double()
    masked_loss = torch.mul(tempMSE,mask)
    return torch.div(torch.sum(masked_loss),num)