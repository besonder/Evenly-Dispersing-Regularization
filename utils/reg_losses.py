# from utils.utils import dc_weights
import torch
from utils.regularizers import ocnn_old, sodso, srip, ad, ph0mst, ocnn



def reg_loss(args, model, fc_weights, kern_weights, conv_weights):
    total_weights = fc_weights + kern_weights + conv_weights
    if args.r == 0:
        return 0
    elif args.reg == 'SO':
        sloss = 0
        for i in range(len(total_weights)):
            sloss += sodso.SO(total_weights[i])
        rloss = args.r*sloss

    elif args.reg == 'DSO':
        sloss = 0
        for i in range(len(total_weights)):
            sloss += sodso.DSO(total_weights[i])
        rloss = args.r*sloss        

    elif args.reg == 'SRIP':
        oloss = srip.l2_reg_ortho(model)
        rloss = args.r*oloss

    elif args.reg == 'OCNN':
        dloss = 0
        closs = 0
        for w in kern_weights:
            dloss += ocnn.orth_dist(w)
        for w, s in conv_weights:
            closs += ocnn.deconv_orth_dist(w, stride=s)     
        rloss = args.r*(dloss + closs)

    elif args.reg == 'ADK':
        Nloss = 0
        Tloss = 0
        if args.fc:
            for i in range(len(fc_weights)):
                nloss, tloss = ad.ADK(fc_weights[i], args.theta, args.double)
                Nloss += nloss
                Tloss += tloss    
        if args.kern:
            for i in range(len(kern_weights)):
                nloss, tloss = ad.ADK(kern_weights[i], args.theta, args.double)
                Nloss += nloss
                Tloss += tloss                          
        if args.conv:
            for i in range(len(conv_weights)):
                nloss, tloss = ad.ADK(conv_weights[i][0], args.theta, args.double)
                Nloss += nloss
                Tloss += tloss 
        rloss = args.r*Nloss + args.r*args.rtn*Tloss

    elif args.reg == 'ADC':
        Nloss = 0
        Tloss = 0
        if args.kern:
            for i in range(len(kern_weights)):
                nloss, tloss = ad.ADK(kern_weights[i], args.theta, args.double)
                Nloss += nloss
                Tloss += tloss

        if args.conv:
            for i, (w, s) in enumerate(conv_weights):
                nloss, tloss = ad.ADC(w, args.theta, stride=s)  
                Nloss += nloss
                Tloss += tloss
        rloss = args.r*Nloss + args.r*args.rtn*Tloss
    
    elif args.reg == 'PH0':
        Nloss = 0
        Tloss = 0
        for i in range(len(conv_weights)):
            nloss, tloss = ph0mst.PH0(conv_weights[i][0])
            Nloss += nloss
            Tloss += tloss
        rloss = args.r*Nloss + args.r*args.rtn*Tloss 

    elif args.reg == 'MST':
        Nloss = 0
        Tloss = 0
        for i in range(len(conv_weights)):
            nloss, tloss = ph0mst.MST(conv_weights[i][0])
            Nloss += nloss
            Tloss += tloss
        rloss = args.r*Nloss + args.r*args.rtn*Tloss

    return rloss 
  