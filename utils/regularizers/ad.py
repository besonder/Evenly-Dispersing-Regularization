import torch
from torch.nn import functional as F
import numpy as np


def ad_function(W, theta, target_norm=1):
    m = W.shape[0]
    WWT = W @ torch.t(W)
    norm2 = torch.diagonal(WWT, 0)
    with torch.no_grad():
        N = (torch.sqrt(norm2[:, None] @ norm2[None, :]).detach() + 1e-8)*1.001
    WWTN = WWT/N
    if theta == 1.5708:
        M = torch.logical_not(torch.eye(m, dtype=bool)).cuda()
        tloss = torch.sum((torch.arccos(WWTN[M]) - theta)**2)
    else:
        Z = torch.logical_not(torch.eye(m, dtype=bool)).cuda()
        M1 = (WWTN > np.cos(theta))*Z
        M2 = (WWTN < -np.cos(theta))*Z
        tloss = torch.sum(((torch.arccos(WWTN[M1]) - theta))**2) + \
            torch.sum((torch.arccos(WWTN[M2]) - 3.1416 + theta)**2)
    
    nloss = torch.sum((target_norm**2 - norm2)**2)
    return nloss, tloss


def ADK(weight, theta, double=False):
    m = weight.shape[0]
    W = weight.view(m, -1)
    n = W.shape[1]

    if double:
        nloss, tloss = ad_function(W, theta, target_norm=1)
        n2, t2 = ad_function(torch.t(W), theta, target_norm=m/n)
        nloss += n2
        tloss += t2       
    else: 
        nloss, tloss = ad_function(W, theta, target_norm=1)
    return nloss, tloss  


def ADC(kernel, theta, stride):
    [o_c, i_c, w, h] = kernel.shape
    padding = int(np.floor((w - 1)/stride)*stride)
    output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    ct = int(np.floor(output.shape[-1]/2))
    norm = torch.sqrt(torch.diagonal(output[:, :, ct, ct], 0)).detach() + 1e-8

    output2 = output / (norm[:, None] @ norm[None, :])[:, :, None, None]

    output2[output2 > 0] -= 1e-3
    output2[output2 < 0] += 1e-3

    output3 = torch.arccos(output2)

    M = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    M[:, : ,ct, ct] = torch.eye(o_c).cuda()
    
    nloss = torch.sum((output[M == 1] - 1)**2)

    if theta == 1.5708:
        tloss = torch.sum((output3[M == 0] - 1.5708)**2)
    else:
        M1 = (output3 < theta)*(M == 0)
        M2 = (output3 > 3.1416 - theta)*(M == 0)
        tloss = torch.sum((output3[M1] - theta)**2) + \
            torch.sum((output3[M2] - 3.1416 + theta)**2)
    return nloss, tloss




    