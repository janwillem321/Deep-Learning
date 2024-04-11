# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:06:08 2024

@author: matthijs leenders

largely copied from:  https://github.com/xueleichen/PSNR-SSIM-UCIQE-UIQM-Python/blob/main/evaluate.py
"""

import numpy as np
from skimage.metrics import structural_similarity
import math
from skimage import color, filters

import torch

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def psnr(a,b):
    #pnsr
    mse = np.mean((a-b)**2)
    psnr = 10*math.log10(255**2/mse)
    return psnr

def ssim(a, b):
    # SSIM calculation with explicit win_size
    ssim = structural_similarity(a, b, multichannel=True, win_size=None, data_range = 1.0, channel_axis = 2)
    return ssim

def uiqm(a):
    rgb = a
    gray = color.rgb2gray(a)

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    #1st term UICM
    rg = rgb[:,:,0] - rgb[:,:,1]
    yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
    rgl = np.sort(rg,axis=None)
    ybl = np.sort(yb,axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = int(al1 * len(rgl))
    T2 = int(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr- uyb) ** 2)

    uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    #2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
    Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
    Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

    Rsobel=np.round(Rsobel).astype(np.uint8)
    Gsobel=np.round(Gsobel).astype(np.uint8)
    Bsobel=np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    #3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm


def eme(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = float(np.min(block))
            blockmax = float(np.max(block))


            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * math.log(blockmax / blockmin)
    return eme

def logamee(ch,blocksize=10):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]
            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            top = plipsub(blockmax,blockmin)
            bottom = plipsum(blockmax,blockmin)

            m = top/bottom
            if m ==0.:
                s+=0
            else:
                s += (m) * np.log(m)

    return plipmult(w,s)

size_test_set = 50

test_set_output = torch.rand(size_test_set, 256, 256, 3)
test_set_truth = torch.rand(size_test_set, 256, 256, 3)

psnr_arr = []
ssim_arr = []
uiqm_arr = []

for i in range(size_test_set):
    
    fake_img = test_set_output[i, :, :, :].numpy()
    gt_img = test_set_truth[i, :, :, :].numpy()
    
    psnr_arr.append(psnr(fake_img, gt_img)) 
    ssim_arr.append(ssim(fake_img, gt_img))
    uiqm_arr.append(uiqm(fake_img))

print(f'PSNR: mean = {np.mean(psnr_arr):.4f}; std = {np.std(psnr_arr):.6f}')
print(f'SSIM: mean = {np.mean(ssim_arr):.4f}; std = {np.std(ssim_arr):.6f}')
print(f'UIQM: mean = {np.mean(uiqm_arr):.4f}; std = {np.std(uiqm_arr):.6f}')