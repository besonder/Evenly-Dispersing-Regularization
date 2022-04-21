import yaml


def load_config(args):
    cfg = yaml.load(open(getattr(args, 'experiment_config'),
                         'r'), Loader=yaml.FullLoader)
    args_dict = {}
    args_dict['model'] = cfg['Model']
    args_dict['data'] = cfg['Dataset']

    cfg_reg = cfg['Regularizer']
    args_dict['reg'] = cfg_reg['name']
    args_dict['fc'] = cfg_reg['fc']
    args_dict['onlyconv'] = cfg_reg['onlyconv']

    cfg_ms = cfg_reg['milestone']
    args_dict['lr_MS'] = cfg_ms['lr_MS']
    args_dict['lr'] = [float(x) for x in cfg_ms['lr']]
    args_dict['reg_MS'] = cfg_ms['reg_MS']
    args_dict['rr'] = cfg_ms['rr']
    args_dict['wr'] = cfg_ms['wr']

    cfg_train = cfg['Train']
    args_dict['rtn'] = float(cfg_train['rtn'])
    args_dict['theta'] = cfg_train['theta']
    args_dict['epochs'] = cfg_train['epochs']
    args_dict['bsize'] = cfg_train['batchsize']
    args_dict['warm'] = cfg_train['warmup']

    for key in args_dict:
        setattr(args, key, args_dict[key])
