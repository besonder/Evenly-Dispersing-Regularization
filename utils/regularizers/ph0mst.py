import torch
import numpy as np
import gudhi as gd
from scipy.sparse.csgraph import minimum_spanning_tree

def PH0(weight, mel=1000):
    m = weight.shape[0]
    W = weight.view(m, -1)
    rips = gd.RipsComplex(W, max_edge_length=mel)
    st = rips.create_simplex_tree(max_dimension=0)
    st.compute_persistence()
    idx = st.flag_persistence_generators()
    if len(idx[0]) == 0:
        verts = torch.empty((0, 2), dtype=int)
    else:
        verts = torch.tensor(idx[0][:, 1:])
    dgm = torch.norm(W[verts[:, 0], :] - W[verts[:, 1], :], dim=-1)
    tloss = torch.sum(dgm)

    norm = torch.norm(W, dim=1)
    nloss = torch.sum((1 - norm**2)**2)
    return nloss, tloss


def MST(weight):
    m = weight.shape[0]
    W = weight.view(m, -1)
    dist = torch.sqrt(torch.sum(torch.pow(W[:, None, :] - W[None, :, :], 2), dim=2))
    Tscr = minimum_spanning_tree(dist.detach().cpu().numpy())
    result = Tscr.toarray()
    mst = np.where(result > 0)
    tloss = torch.sqrt(((W[mst[0]] - W[mst[1]])**2).sum(-1)).sum()

    norm = torch.norm(W, dim=1)
    nloss = torch.sum((1 - norm**2)**2)
    return nloss, tloss