import torch


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


def dc_weights(model):
    down_weights = []
    conv_weights = []

    layer_list = [layer for layer in dir(model) if layer.startswith('layer')]
    for layer in layer_list:
        layer_get = getattr(model, layer)
        for i in range(len(layer_get)):
            try:
                conv = getattr(layer_get[i], 'conv1')
                conv_weights.append((conv.weight, conv.stride[0]))
            except:
                pass
            try:
                down = getattr(layer_get[i], 'downsample')
                down_weights.append(down[0].weight)
            except:
                pass 
    total_weights = down_weights + [w[0] for w in conv_weights]
    return down_weights, conv_weights, total_weights


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

