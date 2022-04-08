import os
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from utils import reg_losses, model_datasets
from utils.utils import AverageMeter, adjust_learning_rate, accuracy, dc_weights


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--data', type=str, default='cifar100')
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--reg', type=str, default='base')
parser.add_argument('--r', type=float, default=0.1)
parser.add_argument('--r_so', type=float, default=0.1)
parser.add_argument('--r_dso', type=float, default=0.1)
parser.add_argument('--r_srip', type=float, default=1e-6)
parser.add_argument('--r_ocnn', type=float, default=0.1)
parser.add_argument('--r_ncad', type=float, default=0.1)
parser.add_argument('--r_tcad', type=float, default=0.01)
parser.add_argument('--theta', type=float, default=1.41)

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--bsize', type=int, default=128)
parser.add_argument('--wdecay', type=bool, default=True)
parser.add_argument('--warm', type=int, default=1)


args = parser.parse_args()

now = datetime.now()
logfile = '_'.join([args.model, args.data, args.reg])
logfile += now.strftime("_%Y-%m-%d_%H-%M-%S") 
logfile += '.txt'

f = open(os.path.join('./log/', logfile), 'w')

for k in args.__dict__:
    f.write(k + ': ' + str(getattr(args, k)) +'\n')

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

model, train_dataset, val_dataset = model_datasets.model_data(args)

model.cuda() 

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

criterion = nn.CrossEntropyLoss().cuda()

regularizer = args.reg

if regularizer == 'base':
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=5e-4
                                )
elif args.wdecay:
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
MILESTONES = [60, 120, 160]

train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2)

down_weights, conv_weights, total_weights = dc_weights(model)


for epoch in range(args.epochs):
    if epoch > args.warm:
        train_scheduler.step()
    # adjust_learning_rate(optimizer, epoch, args.lr)
    
    losses_t = AverageMeter('Loss', ':.4e')
    top1_t = AverageMeter('Acc@1', ':6.2f')
    top5_t = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)

        loss = criterion(output, target) 

        loss += reg_losses.reg_loss(args, down_weights, conv_weights, total_weights, model)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        losses_t.update(loss.item(), images.size(0))
        top1_t.update(acc1[0], images.size(0))
        top5_t.update(acc5[0], images.size(0))
        
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
    
    # print(f'epoch: {epoch}, train loss: {losses_t.avg:.3f}, acc1: {top1_t.avg:.3f}, acc5: {top5_t.avg:.3f}')
    print(f'epoch: {epoch}, validation loss: {losses.avg:.3f}, acc1: {top1.avg:.3f}, acc5: {top5.avg:.3f}')
    # f.write(f'epoch: {epoch}, train loss: {losses_t.avg:.3f}, acc1: {top1_t.avg:.3f}, acc5: {top5_t.avg:.3f}\n')
    f.write(f'epoch: {epoch}, validation loss: {losses.avg:.3f}, acc1: {top1.avg:.3f}, acc5: {top5.avg:.3f}\n')

f.close()



