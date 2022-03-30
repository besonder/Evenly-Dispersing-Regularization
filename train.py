import os
import argparse
import torchvision
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
import ocnn
import srip
import cad
import cad2
import sodso


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int)
parser.add_argument('--reg', type=str, default=None)
parser.add_argument('--r', type=float, default=0.1)
parser.add_argument('--r_so', type=float, default=0.1)
parser.add_argument('--r_dso', type=float, default=0.1)
parser.add_argument('--r_srip', type=float, default=1e-6)
parser.add_argument('--r_ocnn', type=float, default=0.1)
parser.add_argument('--r_ncad', type=float, default=1e-2)
parser.add_argument('--r_tcad', type=float, default=1e-4)
parser.add_argument('--r_ncad2', type=float, default=1e-2)
parser.add_argument('--r_tcad2', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=200)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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
                root='../../DATA/', 
                transform=transforms.Compose(
                    [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]),
                train=True)

val_dataset = torchvision.datasets.CIFAR100(
                root='../../DATA/', 
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
                    batch_size=256, 
                    shuffle=trainsampler is None, 
                    num_workers=4, 
                    pin_memory=True, 
                    sampler=trainsampler
                    )

val_loader = DataLoader(
                    val_dataset, 
                    batch_size=256, 
                    shuffle=False, 
                    num_workers=4, 
                    pin_memory=True, 
                    sampler=validsampler
                    )

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(in_features=512, out_features=100, bias=True)
model.cuda()


criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=args.lr,
                            momentum=0.9,
                            ) # weight_decay=1e-4
regularizer = args.reg

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

if regularizer == 'CAD':
    mycad = cad.CAD(len(target_weights))


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
            diff = ocnn.orth_dist(model.layer2[0].downsample[0].weight) + ocnn.orth_dist(model.layer3[0].downsample[0].weight) + ocnn.orth_dist(model.layer4[0].downsample[0].weight)
            diff += ocnn.deconv_orth_dist(model.layer1[0].conv1.weight, stride=1) + ocnn.deconv_orth_dist(model.layer1[1].conv1.weight, stride=1)
            diff += ocnn.deconv_orth_dist(model.layer2[0].conv1.weight, stride=2) + ocnn.deconv_orth_dist(model.layer2[1].conv1.weight, stride=1)
            diff += ocnn.deconv_orth_dist(model.layer3[0].conv1.weight, stride=2) + ocnn.deconv_orth_dist(model.layer3[1].conv1.weight, stride=1)
            diff += ocnn.deconv_orth_dist(model.layer4[0].conv1.weight, stride=2) + ocnn.deconv_orth_dist(model.layer4[1].conv1.weight, stride=1)        
            loss += args.r_ocnn*diff

        elif regularizer == 'SRIP':
            oloss = srip.l2_reg_ortho(model)
            loss += args.r_srip*oloss

        elif regularizer == 'SO':
            sloss = 0
            for i in range(len(target_weights)):
                sloss += sodso.SO(target_weights[i])
            loss += args.r_so*sloss

        elif regularizer == 'DSO':
            sloss = 0
            for i in range(len(target_weights)):
                sloss += sodso.DSO(target_weights[i])
            loss += args.r_dso*sloss

        elif regularizer == 'CAD':
            Nloss = 0
            Tloss = 0
            for i in range(len(target_weights)):
                # print('shape', target_weights[i].shape)
                nloss, tloss = mycad.loss(i, target_weights[i])
                Nloss += nloss
                Tloss += tloss
            loss += args.r_ncad*Nloss + args.r_tcad*Tloss

        elif regularizer == 'CAD2':
            Nloss = 0
            Tloss = 0
            for i in range(len(target_weights)):
                nloss, tloss = cad2.CAD2(target_weights[i])
                Nloss += nloss
                Tloss += tloss
            loss += args.r_ncad2*Nloss + args.r_tcad2*Tloss

            
        
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


