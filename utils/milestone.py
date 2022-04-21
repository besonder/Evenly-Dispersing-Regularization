def milestones(args):
    if args.data == 'cifar100':
        if args.model.startswith('resnet'):
            args.__dict__['lr_MS'] = [60, 110, 150, 180]
            args.__dict__['lr'] = [1e-1, 2*1e-2, 4*1e-3, 8*1e-4, 2*1e-4]
            args.__dict__['wr'] = [5e-4]

            if args.reg == 'SO':
                args.__dict__['reg_MS'] = [20, 50, 70, 120]
                args.__dict__['rr'] = [1e-1, 1e-3, 1e-4, 1e-6, 0]
                # args.__dict__['wr'] = [1e-8, 5*1e-4, 5*1e-4, 5*1e-4, 5*1e-4]
                args.__dict__['wr'] = [1e-8, 1e-4, 1e-4, 1e-4, 1e-4]

            elif args.reg == 'DSO':
                args.__dict__['reg_MS'] = [20, 50, 70, 120]
                args.__dict__['rr'] = [1e-1, 1e-3, 1e-4, 1e-6, 0]
                args.__dict__['wr'] = [1e-8, 5*1e-4, 5*1e-4, 5*1e-4, 5*1e-4]   

            elif args.reg == 'SRIP':
                args.__dict__['reg_MS'] = [20, 50, 70, 120]
                args.__dict__['rr'] = [1e-3, 1e-4, 1e-5, 1e-6, 0]
                args.__dict__['wr'] = [1e-8, 5*1e-4, 5*1e-4, 5*1e-4, 5*1e-4]

            elif args.reg == 'OCNN':
                args.__dict__['reg_MS'] = [20, 50, 70, 120]
                args.__dict__['rr'] = [1e-1, 1e-3, 1e-4, 1e-6, 0]
                args.__dict__['wr'] = [1e-8, 5*1e-4, 5*1e-4, 5*1e-4, 5*1e-4]

            elif args.reg == 'ADK':
                args.__dict__['reg_MS'] = [20, 50, 70, 120]
                args.__dict__['rr'] = [1e-1, 1e-2, 1e-2, 1e-2, 1e-2]
                # args.__dict__['rr'] = [1e-1, 1e-3, 1e-4, 1e-6, 0]
                args.__dict__['wr'] = [1e-8, 5*1e-4, 5*1e-4, 5*1e-4, 5*1e-4]

            elif args.reg == 'ADC':
                args.__dict__['reg_MS'] = [20, 50, 70, 120]
                args.__dict__['rr'] = [1e-1, 1e-3, 1e-4, 1e-6, 0]
                args.__dict__['wr'] = [1e-8, 5*1e-4, 5*1e-4, 5*1e-4, 5*1e-4]

            elif args.reg == 'PH0' or args.reg == 'MST':
                args.__dict__['reg_MS'] = [20, 50, 70, 120]
                args.__dict__['rr'] = [1e-1, 1e-3, 1e-4, 1e-6, 0]
                args.__dict__['wr'] = [1e-8, 1*1e-4, 1*1e-4, 1*1e-4, 1*1e-4]
