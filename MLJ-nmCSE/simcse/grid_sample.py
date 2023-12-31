import numpy as np
from numpy.linalg import norm
import math
import torch
from torch import nn as nn
from torch.nn import functional as F
import random
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
from scipy.spatial import distance


def dist(u,v):
    return np.linalg.norm(u - v)

def cos_p(t, N, c):
    return np.cos((1 - 1 / 3 * np.sqrt(1 - t * t)) * np.sqrt(N) * c * t)
def sin_p(t,N,c):
    return np.sin((1 - 1 / 3 * np.sqrt(1 - t * t)) * np.sqrt(N) * c * t)

def norm_circle_point(c=8*np.pi,N=128,D=768):
    np.random.seed(42)
    ts = 2 * np.linspace(0, N - 1, N) / (N - 1) - 1
    points = []

    for t in ts:
        x1 = t
        x2 = np.sqrt(1 - t * t) * cos_p(t, N, c)
        temp_x = x1 * x1 + x2 * x2
        x = [x1, x2]

        for d in range(3,D):
            x_d = np.sqrt(1-temp_x) * cos_p(t,N,c)
            temp_x += x_d * x_d
            x.append(x_d)

        if D == 3:
            temp_x = x1 * x1

        if D >= 3:
            x_last = np.sqrt(1-temp_x) * sin_p(t,N,c)
            x.append(x_last)

        points.append(x)

    return np.array(points)


def uniform_circle_point(D=768, N=128):
    np.random.seed(42)
    X = np.random.default_rng().normal(size=(N, D))

    return 1 / np.sqrt(np.sum(X ** 2, 1, keepdims=True)) * X

def uniform_loss(x, t=2):
    z = F.normalize(x,dim=1)
    return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log()

def cos_m(x):
    if type(x) != torch.tensor:
        x = torch.tensor(x)
    cos = nn.CosineSimilarity(dim=-1)
    value = cos(x.unsqueeze(1),x.unsqueeze(0))
    return value

def Sample_By_Cos(neg_f, n):

    value = cos_m(neg_f).sum(axis=1)
    idx = torch.topk(value,int(n))[1]
    return idx.detach().cpu()


def Grid_Sample(neg_f, circle_p, dis_neg_cir, n, r=1):
    random.seed(42)
    if neg_f.device.type == 'cuda':
        neg_f = neg_f.detach().cpu().numpy()
    
    n = min(n,neg_f.shape[0])

    neg_p_now = dis_neg_cir.argmin()

    norm_neg_p = neg_f * (1/dis_neg_cir).repeat(neg_f.shape[1],axis=0).reshape(neg_f.shape[0],-1)
    norm_dis_neg_cir = distance.cdist(norm_neg_p, circle_p)

    sampled_neg_p = [neg_p_now]
    cir_now = norm_dis_neg_cir[neg_p_now].argmin()
    sampled_circle = [cir_now]

    for i in range(n-1):
        temp_list = list(set(range(len(circle_p))) - set(sampled_circle))
        new_cir_now = random.choice(temp_list)
        for j in range(2 * len(temp_list)):
            new_cir_now = random.choice(temp_list)
            if dist(circle_p[new_cir_now],circle_p[cir_now]) >= r:
                break
        cir_now = new_cir_now
        sampled_circle.append(cir_now)
        neg_p_now = norm_dis_neg_cir[:,cir_now].argmin()
        sampled_neg_p.append(neg_p_now)

    return sampled_neg_p



