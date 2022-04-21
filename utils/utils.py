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



def reg_weights(model):
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

