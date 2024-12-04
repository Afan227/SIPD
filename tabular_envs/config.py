from numpy import sqrt
import torch
import os
import math
working_dir = '.'
import torch.nn as nn
ITER_UPPER_BOUND = 10000
SAMPLE_NUM_POINT = 100
LR_COEFF = 1. * sqrt(ITER_UPPER_BOUND)
ETA = 0.013
SAMPLE_Y_SIZE = 100
TRAJ_NUM = 100000
TRAJ_LEN = 100
TRAIN_TRAJ_NUM = 1000
TRAIN_TRAJ_LEN = 100
INNER_INIT_LR = 1.
DEFAULT_W = 1000  # 100

PI_MODE = '100'  # mean, last or str(number)

SEED = 74751

GAMMA = 0.9

CHECK_FINENESS = 1000

SEPARATION_LEN = 50

# def get_device(gpu: int) -> torch.device:
#     if gpu == -1:
#         os.environ['CUDA_VISIBLE_DEVICES'] = ''
#         return torch.device('cpu')
#     else:
#         return torch.device(f'cuda:{gpu}')
#
# pi_logit = torch.zeros((8, 5), device=torch.device('cpu'))
# print(pi_logit[0,1])
#
# softmax = nn.Softmax(dim=pi_logit.ndim - 1)
# print(softmax(pi_logit))a =

# a = torch.tensor([1,2,3,4])
# b = torch.tensor([1,2,3,4])
# c = torch.tensor([[1,2,3,4,5],[5,4,3,2,1],[9,8,7,6,5],[8,7,6,5,4],[11,12,13,14,15]])
# d = c[a,b]
# print(d)
# device = torch.device('cpu')
# dim_Y = 2
# lb_Y = torch.zeros((dim_Y), device=device)
# ub_Y = torch.ones((dim_Y), device=device) * 2.
# print(lb_Y)
# print(ub_Y)
# x = torch.sum(ub_Y)
# print(x)