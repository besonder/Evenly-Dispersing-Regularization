import torch


def SO(weight):
    if len(weight) == 2:
        weight = weight[0]
    m = weight.shape[0]
    W = weight.view(m, -1)
    loss = torch.sum((W @ torch.t(W) - torch.eye(m, dtype=float).cuda())**2)
    return loss


def DSO(weight):
    if len(weight) == 2:
        weight = weight[0]
    m = weight.shape[0]
    W = weight.view(m, -1)
    n = W.shape[1]
    loss = torch.sum((W @ torch.t(W) - torch.eye(m, dtype=float).cuda())**2) + \
        torch.sum((torch.t(W) @ W - torch.eye(n, dtype=float).cuda())**2) 
    return loss
