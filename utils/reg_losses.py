from utils.utils import dc_weights
from utils.regularizers import sodso, srip, ocnn, cad


def reg_loss(args, down_weights, conv_weights, total_weights, model):
    rloss = 0
    if args.reg == 'OCNN':
        dloss = 0
        closs = 0
        for w in down_weights:
            dloss += ocnn.orth_dist(w)
        for w, s in conv_weights:
            closs += ocnn.deconv_orth_dist(w, stride=s)       
        rloss += args.r_ocnn*(dloss + closs)

    elif args.reg == 'SRIP':
        oloss = srip.l2_reg_ortho(model)
        loss += args.r_srip*oloss

    elif args.reg == 'SO':
        sloss = 0
        for i in range(len(total_weights)):
            sloss += sodso.SO(total_weights[i])
        rloss += args.r_so*sloss

    elif args.reg == 'DSO':
        sloss = 0
        for i in range(len(total_weights)):
            sloss += sodso.DSO(total_weights[i])
        rloss += args.r_dso*sloss        

    elif args.reg == 'CAD':
        Nloss = 0
        Tloss = 0
        for i in range(len(total_weights)):
            nloss, tloss = cad.CAD(total_weights[i], args.theta)
            Nloss += nloss
            Tloss += tloss
        rloss += args.r_ncad*Nloss + args.r_tcad*Tloss 

    elif args.reg == 'CAD2':
        nloss = 0
        tloss = 0
        for w in down_weights:
            nl, tl = cad.CAD(w, args.theta)
            nloss += nl
            tloss += tl
        for w, s in conv_weights:
            nl, tl = cad.deconv_orth_dist(w, stride=s)       
            nloss += nl
            tloss += tl
        rloss += args.r_ncad*nloss + args.r_tcad*tloss

    return rloss
