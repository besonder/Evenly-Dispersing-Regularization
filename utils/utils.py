import torch
import random
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res          


def adjust_learning_rate(optimizer, epoch, args):
    i = 0
    j = 0
    if args.reg == 'base':
        while epoch >= args.lr_MS[i]:
            i += 1
            if i == len(args.lr_MS):
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr[i]
    else:
        while epoch >= args.lr_MS[i]:
            i += 1
            if i == len(args.lr_MS):
                break
        while epoch >= args.reg_MS[j]:
            j += 1
            if j == len(args.reg_MS):
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr[i]
            param_group['weight_decay'] = args.wr[j]
        args.r = args.rr[j]



def reg_weights(model, args):
    first_conv = True
    fc_weights = []
    kern_weights = []
    conv_weights = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.kernel_size[0] == 1:
                kern_weights.append(m.weight)
            else:
                if first_conv:
                    kern_weights.append(m.weight)
                    first_conv = False
                else:
                    conv_weights.append((m.weight, m.stride[0]))

        elif isinstance(m, torch.nn.Linear):
            fc_weights.append(m.weight)
    return fc_weights, kern_weights, conv_weights


def do_seed(seed_num, cudnn_ok=True):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)  # if use multi-GPU
    # It could be slow
    if cudnn_ok:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def angle_analysis(weight):
    m = weight.shape[0]
    if len(weight.shape) == 4:
        weight = weight.view(m, -1)
    WWT = weight @ torch.t(weight)
    norm2 = torch.diagonal(WWT, 0)
    N = (torch.sqrt(norm2[:, None] @ norm2[None, :]) + 1e-8)*1.001
    WWTN = WWT/N

    M = torch.logical_not(torch.eye(m))
    sp = torch.sort(1 - torch.abs(WWTN[M].view(m, -1)), dim=1)
    
    theta = torch.arccos(torch.abs(WWTN[torch.arange(m), sp.indices[:, 0]]))
    mean = torch.mean(theta)
    Max = torch.amax(theta)
    Min = torch.amin(theta)
    meanNorm = torch.mean(torch.norm(weight, dim=1))
    return mean, Min, Max, meanNorm


def weights_angle_analysis(fc_weights, kern_weights, conv_weights, f, view=-1):
    total_weights = fc_weights + kern_weights + [ws[0] for ws in conv_weights]
    with torch.no_grad():
        for i, W in enumerate(total_weights):
            if view == -1 or i == view:
                print_result = True
            if print_result:
                mean, Min, Max, meanNorm = angle_analysis(W)
                print(f'weight {i}, shape: {W.shape}, mean: {mean:.3f}, Min: {Min:.3f}, Max: {Max:.3f}, Norm: {meanNorm:.3f}')
                f.write(f'weight {i}, shape: {W.shape}, mean: {mean:.3f}, Min: {Min:.3f}, Max: {Max:.3f}, Norm: {meanNorm:.3f}\n')
                print_result = False



