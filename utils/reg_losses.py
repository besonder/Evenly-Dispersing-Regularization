from utils.utils import dc_weights
from utils.regularizers import sodso, srip, ocnn, ad


def reg_loss(args, down_weights, conv_weights, total_weights, model):
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

    elif args.reg == 'MC':
        pass

    elif args.reg == 'SRIP':
        oloss = srip.l2_reg_ortho(model)
        rloss = args.r*oloss

    elif args.reg == 'OCNN':
        dloss = 0
        closs = 0
        for w in down_weights:
            dloss += ocnn.orth_dist(w)
        for w, s in conv_weights:
            closs += ocnn.deconv_orth_dist(w, stride=s)     
        rloss = args.r*(dloss + closs)

    elif args.reg == 'ADK':
        Nloss = 0
        Tloss = 0
        for i in range(len(total_weights)):
            nloss, tloss = ad.ADK(total_weights[i], args.theta)
            Nloss += nloss
            Tloss += tloss
        rloss = args.r*Nloss + args.r*args.rtn*Tloss 

    elif args.reg == 'ADC':
        nloss = 0
        tloss = 0
        for w in down_weights:
            nl, tl = ad.ADK(w, args.theta)
            nloss += nl
            tloss += tl
        for w, s in conv_weights:
            nl, tl = ad.ADC(w, args.theta, stride=s)       
            nloss += nl
            tloss += tl
        rloss = args.r*nloss + args.r*args.rtn*tloss
    return rloss
