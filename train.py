import os
import argparse
import torchvision
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
import sodso
import ocnn
import srip
import cad



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--reg', type=str, default=None)
parser.add_argument('--r', type=float, default=0.1)
parser.add_argument('--r_so', type=float, default=0.1)
parser.add_argument('--r_dso', type=float, default=0.1)
parser.add_argument('--r_srip', type=float, default=1e-6)
parser.add_argument('--r_ocnn', type=float, default=0.1)
parser.add_argument('--r_ncad', type=float, default=0.1)
parser.add_argument('--r_tcad', type=float, default=0.01)
parser.add_argument('--r_cad2', type=float, default=0.01)
parser.add_argument('--theta', type=float, default=1.41)

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--bsize', type=int, default=256)
parser.add_argument('--wdecay', action='store_true')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if "WORLD_SIZE" in os.environ:
    world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = world_size > 1
    args.world_size = world_size
else:
    args.distributed = False
    args.world_size = 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

# do_seed(args.seed)


if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')


train_dataset = torchvision.datasets.CIFAR100(
                root='./DATA/', 
                transform=transforms.Compose(
                    [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]),
                train=True)

val_dataset = torchvision.datasets.CIFAR100(
                root='./DATA/', 
                transform=transforms.Compose(
                    [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]),
                train=False)

if args.distributed:
    trainsampler = DistributedSampler(train_dataset)
    validsampler = DistributedSampler(val_dataset)
else:
    trainsampler = None
    validsampler = None


train_loader = DataLoader(
                    train_dataset, 
                    batch_size=args.bsize, 
                    shuffle=trainsampler is None, 
                    num_workers=4, 
                    pin_memory=True, 
                    sampler=trainsampler
                    )

val_loader = DataLoader(
                    val_dataset, 
                    batch_size=args.bsize, 
                    shuffle=False, 
                    num_workers=4, 
                    pin_memory=True, 
                    sampler=validsampler
                    )

if args.model == 'resnet18':
    model = torchvision.models.resnet18()
elif args.model == 'resnet34':
    model = torchvision.models.resnet34()
elif args.model == 'resnet50':
    model = torchvision.models.resnet50()
elif args.model == 'resnet101':
    model = torchvision.models.resnet101()
elif args.model == 'resnet152':
    model = torchvision.models.resnet152()


model.fc = torch.nn.Linear(in_features=512, out_features=100, bias=True)
model.cuda()


criterion = nn.CrossEntropyLoss().cuda()

regularizer = args.reg

if (regularizer is None) or args.wdecay:
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=1e-4
                                )
else:
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr,
                                momentum=0.9,
                                ) 

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
    
    
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


target_weights = [
            model.layer2[0].downsample[0].weight, 
            model.layer3[0].downsample[0].weight,
            model.layer4[0].downsample[0].weight,
            model.layer1[0].conv1.weight,
            model.layer1[1].conv1.weight,
            model.layer2[0].conv1.weight,
            model.layer2[1].conv1.weight,
            model.layer3[0].conv1.weight,
            model.layer3[1].conv1.weight,
            model.layer4[0].conv1.weight,
            model.layer4[1].conv1.weight,
        ]

down_weights = []
conv_weights = []
total_weights = [down_weights] + [w[0] for w in conv_weights]


layer_list = [layer for layer in dir(model) if layer.startswith('layer')]
for layer in layer_list:
    layer_get = getattr(model, layer)
    for i in range(len(layer_get)):
        try:
            conv = getattr(layer_get[i], 'conv1')
            conv_weights.append((conv.weight, conv.stride))
        except:
            pass
        try:
            down = getattr(layer_get[i], 'downsample')
            down_weights.append(down[0].weight)
        except:
            pass 


for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch, args.lr)
    
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)

        loss = criterion(output, target) 

        if regularizer == 'OCNN':
            dloss = 0
            closs = 0
            for w in down_weights:
                dloss += ocnn.orth_dist(w)
            for w, s in conv_weights:
                closs += ocnn.deconv_orth_dist(w, stride=s)       
            loss += args.r_ocnn*(dloss + closs)

        elif regularizer == 'SRIP':
            oloss = srip.l2_reg_ortho(model)
            loss += args.r_srip*oloss

        elif regularizer == 'SO':
            sloss = 0
            for i in range(len(total_weights)):
                sloss += sodso.SO(total_weights[i])
            loss += args.r_so*sloss

        elif regularizer == 'DSO':
            sloss = 0
            for i in range(len(total_weights)):
                sloss += sodso.DSO(total_weights[i])
            loss += args.r_dso*sloss        

        elif regularizer == 'CAD':
            Nloss = 0
            Tloss = 0
            for i in range(len(total_weights)):
                nloss, tloss = cad.CAD(total_weights[i], args.theta)
                Nloss += nloss
                Tloss += tloss
            loss += args.r_ncad*Nloss + args.r_tcad*Tloss 

        elif regularizer == 'CAD2':
            dloss = 0
            nloss = 0
            tloss = 0
            for w in down_weights:
                dloss += cad.orth_dist(w)
            for w, s in conv_weights:
                nl, tl = cad.deconv_orth_dist(w, stride=s)       
                nloss += nl
                tloss += tl
            loss += args.r_cad2*(dloss + nloss + tloss)


        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    ####### validataion #######
    
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.eval()
    for i, (images, target) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)

        loss = criterion(output, target) 
        acc1, acc5 = accuracy(output, target, topk=(1, 5))    
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
    print(f'epoch: {epoch}, validation loss: {losses.avg:.3f}, acc1: {top1.avg:.3f}, acc5: {top5.avg:.3f}')


