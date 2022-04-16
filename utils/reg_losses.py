# from utils.utils import dc_weights
import torch
from utils.regularizers import ocnn_old, sodso, srip, ad, ph0mst, ocnn



def reg_loss(args, kern_weights, conv_weights, model):
    total_weights = kern_weights + conv_weights
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
        for i in range(len(kern_weights)):
            nloss, tloss = ad.ADK_fc(kern_weights[i], args.theta)
            Nloss += nloss
            Tloss += tloss
        for i in range(len(conv_weights)):
            nloss, tloss = ad.ADK(conv_weights[i], args.theta)
            Nloss += nloss
            Tloss += tloss
        rloss = args.r*Nloss + args.r*args.rtn*Tloss 

    elif args.reg == 'ADC':
        Nloss = 0
        Tloss = 0
        for i in range(len(kern_weights)):
            nloss, tloss = ad.ADK_fc(kern_weights[i], args.theta)
            Nloss += nloss
            Tloss += tloss

        for i, (w, s) in enumerate(conv_weights):
            nloss, tloss = ad.ADC(w, args.theta, stride=s)  

            # m = w.shape[0]
            # A = w.view(m, -1)     
            # E = torch.mean(torch.sqrt(torch.sum(A**2, dim=1)))
            # print(i, w.shape, E.item(), nloss.item(), tloss.item())

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


# def reg_loss(args, down_weights, conv_weights, total_weights, model):
#     if args.r == 0:
#         return 0
#     elif args.reg == 'SO':
#         sloss = 0
#         for i in range(len(total_weights)):
#             sloss += sodso.SO(total_weights[i])
#         rloss = args.r*sloss

#     elif args.reg == 'DSO':
#         sloss = 0
#         for i in range(len(total_weights)):
#             sloss += sodso.DSO(total_weights[i])
#         rloss = args.r*sloss        

#     elif args.reg == 'SRIP':
#         oloss = srip.l2_reg_ortho(model)
#         rloss = args.r*oloss

#     elif args.reg == 'OCNN':
#         dloss = 0
#         closs = 0
#         for w in down_weights:
#             dloss += ocnn.orth_dist(w)
#         for w, s in conv_weights:
#             closs += ocnn.deconv_orth_dist(w, stride=s)     
#         rloss = args.r*(dloss + closs)

#     elif args.reg == 'ADK':
#         Nloss = 0
#         Tloss = 0
#         for i in range(len(total_weights)):
#             nloss, tloss = ad.ADK(total_weights[i], args.theta)
#             Nloss += nloss
#             Tloss += tloss
#         rloss = args.r*Nloss + args.r*args.rtn*Tloss 

#     elif args.reg == 'ADC':
#         nloss = 0
#         tloss = 0
#         print(len(down_weights), len(conv_weights))
#         for w in down_weights:
#             nl, tl = ad.ADK(w, args.theta)
#             print('##', nl, tl)
#             nloss += nl
#             tloss += tl

#         print(model.fc.weight.shape, args.theta)
#         nl, tl = ad.ADK_fc(model.fc.weight, args.theta)
#         print(nloss, nl)
#         nloss += nl
#         tloss += tl

#         for w, s in conv_weights:
#             nl, tl = ad.ADC(w, args.theta, stride=s)       
#             nloss += nl
#             tloss += tl
#         rloss = args.r*nloss + args.r*args.rtn*tloss
    
#     elif args.reg == 'PH0':
#         Nloss = 0
#         Tloss = 0
#         for i in range(len(total_weights)):
#             nloss, tloss = ph0mst.PH0(total_weights[i], args.theta)
#             Nloss += nloss
#             Tloss += tloss
#         rloss = args.r*Nloss + args.r*args.rtn*Tloss 

#     elif args.reg == 'MST':
#         Nloss = 0
#         Tloss = 0
#         for i in range(len(total_weights)):
#             nloss, tloss = ph0mst.MST(total_weights[i], args.theta)
#             Nloss += nloss
#             Tloss += tloss
#         rloss = args.r*Nloss + args.r*args.rtn*Tloss

#     return rloss      