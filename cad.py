import torch
import numpy as np


class CAD():
    def __init__(self, numweights):
        self.thres_cal = False
        self.numweights = numweights
        self.thres_list = []

    def loss(self, idx, weights):
        W = weights.view(weights.shape[0], -1)
        m, n = W.shape
        delta = self.get_delta(idx, m, n)
        delta = 0.99

        cosd = np.cos(delta)
        I = torch.eye(m, dtype=float).cuda()
        WWT = W@torch.t(W)
        nloss = torch.sum((WWT * I - I)**2)

        M = (torch.abs(WWT.detach()) < delta).type(WWT.dtype)*(1 - I)
        tloss = torch.sum(((torch.arccos((torch.abs(WWT) - delta)*M))*M)**2)
        return nloss, tloss


    def get_delta(self, idx, m, n):
        if self.thres_cal is False:
            delta = self.delta_cal(m, n)
            self.thres_list.append(delta)
            if idx == self.numweights - 1:
                self.thres_cal = True
        else:
            delta = self.thres_list[idx] 
        return delta       
    
    def delta_cal(self, m, n):
        sv = self.SV(n)
        d = 2*np.power(sv/(2*m), 1/n)
        # print(d, sv, m, n)
        return d

    def SV(self, n):
        if n//2 == 0:
            k = 2
            sv = 4 * 3.141592
            while k + 2 <= n:
                k += 2
                sv *= k/(k-1)
        else:
            k = 1
            sv = 3.141592
            while k + 2 <= n:
                k += 2
                sv *= k/(k-1)
        return sv
