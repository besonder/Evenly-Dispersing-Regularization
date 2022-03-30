import torch


def CAD2(weight):
    m = weight.shape[0]
    W = weight.view(m, -1)
    WWT = W @ torch.t(W)
    norm = torch.diagonal(WWT, 0) 
    N = torch.sqrt(norm[:, None] @ norm[None, :])
    M = (WWT / N < 1.41)*(1 - torch.eye(m, dtype=float).cuda())
    tloss = torch.sum(((1.41 - WWT*M)*M)**2)
    nloss = torch.sum((1 - norm)**2)
    return nloss, tloss