import os
import sys
import copy
from time import time
import argparse
import numpy as np
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
# import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from utils import models_datasets, reg_losses
from utils.utils import AverageMeter, adjust_learning_rate, accuracy, reg_weights, do_seed, weights_angle_analysis
from utils.config import load_config
# from utils.milestone import milestones


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_config', '-e', type=str,
                    default='configs/resnet18_cifar10_SRIP.yml')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--num_worker', type=int, default=4)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--angle_view', '-av', type=int, default=None)

args = parser.parse_args()

for i, arg in enumerate(sys.argv[1:]):
    if "--local_rank" in arg:
        args.__setattr__("local_rank", int(arg[-1]))
    # elif "--" in arg:
    #     args.__setattr__(arg[2:], str(sys.argv[1:][i+1]))

load_config(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['OMP_NUM_THREADS'] = str(4)
os.environ['MKL_NUM_THREADS'] = str(4)


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

do_seed(args.seed)

model, train_dataset, val_dataset = models_datasets.model_data(args)

model.cuda()

if args.distributed:
    model = DDP(model, device_ids=[args.local_rank],
                output_device=args.local_rank)
    trainsampler = DistributedSampler(train_dataset)
    validsampler = DistributedSampler(val_dataset)
else:
    trainsampler = None
    validsampler = None

if not args.distributed or args.local_rank == 0:
    now = datetime.now()
    logfile = '_'.join([args.model, args.data, args.reg])
    logfile += now.strftime("_%Y-%m-%d_%H-%M-%S")
    logfile += '.txt'

    f = open(os.path.join(args.log, logfile), 'w')

    for k in args.__dict__:
        print(k + ': ' + str(getattr(args, k)))
        f.write(k + ': ' + str(getattr(args, k)) + '\n')

train_loader = DataLoader(
    train_dataset,
    batch_size=args.bsize,
    shuffle=trainsampler is None,
    num_workers=args.num_worker,
    pin_memory=True,
    sampler=trainsampler
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.bsize,
    shuffle=False,
    num_workers=args.num_worker,
    pin_memory=True,
    sampler=validsampler
)

criterion = nn.CrossEntropyLoss().cuda()

regularizer = args.reg

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr[0],
        momentum=0.9,
        weight_decay=args.wr[0]
    )
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr[0],
        weight_decay=args.wr[0]
    )


fc_weights, kern_weights, conv_weights = reg_weights(model, args)

time_t = AverageMeter('Time', ':6.2f')

save_weights = []

for epoch in range(args.epochs):

    adjust_learning_rate(optimizer, epoch, args)

    losses_t = AverageMeter('Loss', ':.4e')
    top1_t = AverageMeter('Acc@1', ':6.2f')
    top5_t = AverageMeter('Acc@5', ':6.2f')

    stime = time()
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)

        loss = criterion(output, target)

        if args.reg != 'base':
            loss += reg_losses.reg_loss(args, model, fc_weights, kern_weights, conv_weights)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses_t.update(loss.item(), images.size(0))
        top1_t.update(acc1[0], images.size(0))
        top5_t.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    time_t.update(time()-stime)

    if args.angle_view is not None:
        weights_angle_analysis(fc_weights, kern_weights, conv_weights, f, args.angle_view)

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
    if not args.distributed or args.local_rank == 0:
        print(f'epoch: {epoch}, tr_time: {time_t.val:.1f}, validation loss: {losses.avg:.3f}, acc1: {top1.avg:.3f}, acc5: {top5.avg:.3f}')
        f.write(
            f'epoch: {epoch}, tr_time: {time_t.val:.1f}, validation loss: {losses.avg:.3f}, acc1: {top1.avg:.3f}, acc5: {top5.avg:.3f}\n')
    

    result = f'epoch_{epoch}_top1_{top1.avg:.3f}'

    
    weight_name = '_'.join([args.model, args.data, args.reg])
    weight_name += now.strftime("_%Y-%m-%d_%H-%M-%S")
    weight_name += '_' + result
    path = weight_name + '.pt'
    path = os.path.join('model_weights', path)

    saveW = False

    if len(save_weights) < 1:
        save_weights.append(weight_name)
        saveW = True
    else:
        saveW = False
        for i, et in enumerate(save_weights):
            if float(et.split('_')[-1]) < top1.avg:
                remove_file = copy.deepcopy(save_weights[i])
                save_weights[i] = weight_name
                saveW= True
                os.remove(os.path.join('model_weights', remove_file+'.pt'))
                break

    if len(save_weights) > 1:
        top5model_loss = np.array([float(et.split('_')[-1]) for et in save_weights])
        sort_idx = np.argsort(top5model_loss)
        save_weights = [save_weights[i] for i in sort_idx]


    if saveW:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': losses.avg,
            'top1': top1.avg,
        }, path)


if not args.distributed or args.local_rank == 0:
    f.close()
