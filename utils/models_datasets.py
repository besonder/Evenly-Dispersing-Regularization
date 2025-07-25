import numpy as np
import torchvision
import torchvision.transforms as transforms
from models import resnet_cifar10, resnet_cifar100, wide_resnet


def model_data(args):

    if args.data == 'cifar100':
        if args.model == 'resnet18':
            model = resnet_cifar100.resnet18()
        elif args.model == 'resnet34':
            model = resnet_cifar100.resnet34()
        elif args.model == 'resnet50':
            model = resnet_cifar100.resnet50()
        elif args.model == 'resnet101':
            model = resnet_cifar100.resnet101()
        elif args.model == 'resnet152':
            model = resnet_cifar100.resnet152()
        elif args.model == 'wideResnet':
            model = wide_resnet.Wide_ResNet(28, 10, 0.3, 100)

        train_dataset = torchvision.datasets.CIFAR100(
                        root='./DATA/', 
                        transform=transforms.Compose(
                            [
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                            ]),
                        train=True)

        val_dataset = torchvision.datasets.CIFAR100(
                        root='./DATA/', 
                        transform=transforms.Compose(
                            [
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                            ]),
                        train=False)

    elif args.data == 'cifar10':
        if args.model == 'resnet18':
            model = resnet_cifar10.ResNet18()
        elif args.model == 'resnet34':
            model = resnet_cifar10.ResNet34()
        elif args.model == 'resnet50':
            model = resnet_cifar10.ResNet50()
        elif args.model == 'resnet101':
            model = resnet_cifar10.ResNet101()
        elif args.model == 'resnet152':
            model = resnet_cifar10.ResNet152()
        elif args.model == 'wideResnet':
            model = wide_resnet.Wide_ResNet(28, 10, 0.3, 10)
            
        train_dataset = torchvision.datasets.CIFAR10(
                        root='./DATA/', 
                        transform=transforms.Compose(
                            [
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                            ]),
                        train=True)

        val_dataset = torchvision.datasets.CIFAR10(
                        root='./DATA/', 
                        transform=transforms.Compose(
                            [
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                            ]),
                        train=False) 

    elif args.data == 'svhn':
        if args.model == 'resnet18':
            model = resnet_cifar10.ResNet18()
        elif args.model == 'resnet34':
            model = resnet_cifar10.ResNet34()
        elif args.model == 'resnet50':
            model = resnet_cifar10.ResNet50()
        elif args.model == 'resnet101':
            model = resnet_cifar10.ResNet101()
        elif args.model == 'resnet152':
            model = resnet_cifar10.ResNet152()
        elif args.model == 'wideResnet':
            model = wide_resnet.Wide_ResNet(28, 10, 0.3, 10)
            
        train_dataset = torchvision.datasets.SVHN(
                        root='./DATA/SVHN/', 
                        transform=transforms.Compose(
                            [
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            transforms.Normalize((109.9/255, 109.7/255, 113.8/255), (50.1/255, 50.6/255, 50.8/255))
                            ]),
                        split='train')

        # extra_dataset = torchvision.datasets.SVHN(
        #                 root='./DATA/SVHN/', 
        #                 transform=transforms.Compose(
        #                     [
        #                     transforms.RandomCrop(32, padding=4),
        #                     transforms.RandomHorizontalFlip(),
        #                     transforms.RandomRotation(15),
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((109.9/255, 109.7/255, 113.8/255), (50.1/255, 50.6/255, 50.8/255))
        #                     ]),
        #                 split='extra')

        val_dataset = torchvision.datasets.SVHN(
                        root='./DATA/SVHN/', 
                        transform=transforms.Compose(
                            [
                            transforms.ToTensor(),
                            transforms.Normalize((109.9/255, 109.7/255, 113.8/255), (50.1/255, 50.6/255, 50.8/255))
                            ]),
                        split='test') 

        # data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        # labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
        # train_dataset.data = data
        # train_dataset.labels = labels


    # elif args.data == 'imagenet':
    #     if args.model == 'resnet18':
    #         model = torchvision.models.resnet18()
    #     elif args.model == 'resnet34':
    #         model = torchvision.models.resnet34()
    #     elif args.model == 'resnet50':
    #         model = torchvision.models.resnet50()
    #     elif args.model == 'resnet101':
    #         model = torchvision.models.resnet101()
    #     elif args.model == 'resnet152':
    #         model = torchvision.models.resnet152()

    #     train_dataset = torchvision.datasets.ImageFolder(
    #                     root='./DATA/', 
    #                     transform=transforms.Compose(
    #                         [
    #                         transforms.RandomResizedCrop(224),
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #                         ]),
    #                     )

    #     val_dataset = torchvision.datasets.ImageFolder(
    #                     root='./DATA/', 
    #                     transform=transforms.Compose(
    #                         [
    #                         transforms.Resize(256),
    #                         transforms.CenterCrop(224),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #                         ]),
    #                     )    
    return model, train_dataset, val_dataset