# from utils.utils import dc_weights
import torch
from utils.regularizers import ocnn_old, sodso, srip, ad, ph0mst, ocnn



def reg_loss(args, model, fc_weights, kern_weights, conv_weights):
    if args.r == 0:
        return 0
    elif args.reg == 'SO':
        sloss = 0

        if not args.fc:
            for i in range(len(fc_weights)):
                sloss += sodso.SO(fc_weights[i])
        if not args.kern:
            for i in range(len(kern_weights)):
                sloss += sodso.SO(kern_weights[i])
        if not args.conv:
            for i in range(len(conv_weights)):
                sloss += sodso.SO(conv_weights[i])
        rloss = args.r*sloss

    elif args.reg == 'DSO':
        sloss = 0
        if not args.fc:
            for i in range(len(fc_weights)):
                sloss += sodso.DSO(fc_weights[i])
        if not args.kern:
            for i in range(len(kern_weights)):
                sloss += sodso.DSO(kern_weights[i])
        if not args.conv:
            for i in range(len(conv_weights)):
                sloss += sodso.DSO(conv_weights[i])
        rloss = args.r*sloss        

    elif args.reg == 'SRIP':
        sloss = 0
        if args.kern:
            for w in kern_weights:
                sloss += srip.l2_reg_ortho(w)
        if args.conv:
            for w, s in conv_weights:
                sloss += srip.l2_reg_ortho(w)
        rloss = args.r*sloss

    elif args.reg == 'OCNN':
        dloss = 0
        closs = 0
        if args.kern:
            for w in kern_weights:
                dloss += ocnn.orth_dist(w)
        if args.conv:
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
            for i, (w, s) in enumerate(conv_weights):
                nloss, tloss = ad.ADC(w, args.theta, stride=s)  
                Nloss += nloss
                Tloss += tloss
        rloss = args.r*Nloss + args.r*args.rtn*Tloss
    
    elif args.reg == 'PH0':
        Nloss = 0
        Tloss = 0
        if args.fc:
            for i in range(len(fc_weights)):
                nloss, tloss = ph0mst.PH0(fc_weights[i])
                Nloss += nloss
                Tloss += tloss
        if args.kern:
            for i in range(len(kern_weights)):
                nloss, tloss = ph0mst.PH0(kern_weights[i])
                Nloss += nloss
                Tloss += tloss  
        if args.conv:      
            for i in range(len(conv_weights)):
                nloss, tloss = ph0mst.PH0(conv_weights[i])
                Nloss += nloss
                Tloss += tloss
        rloss = args.r*Nloss + args.r*args.rtn*Tloss 

    elif args.reg == 'MST':
        Nloss = 0
        Tloss = 0
        if args.fc:
            for i in range(len(fc_weights)):
                nloss, tloss = ph0mst.MST(fc_weights[i])
                Nloss += nloss
                Tloss += tloss
        if args.kern:
            for i in range(len(kern_weights)):
                nloss, tloss = ph0mst.MST(kern_weights[i])
                Nloss += nloss
                Tloss += tloss  
        if args.conv:      
            for i in range(len(conv_weights)):
                nloss, tloss = ph0mst.MST(conv_weights[i])
                Nloss += nloss
                Tloss += tloss
        rloss = args.r*Nloss + args.r*args.rtn*Tloss

    return rloss 
  