
from .attacks import FGSM, PGD

def get_cfg(args):
    if 'cifar' in args.dataset :
        return CIFARconfig
    elif 'mnist' in args.dataset :
        return MNISTconfig
    else :
        return MEDICALconfig



class CIFARconfig:
    method_and_iter = [ ( FGSM, 0 ), ( PGD, 7 ), ( PGD, 20 ) ]
    eps_list = list(range(1, 20))
    eps_div = 255
    alpha_amp = 1/4

    in_channels = 3
    out_channels = 3
    depth = 4

class MNISTconfig:
    method_and_iter = [ ( FGSM, 0 ), ( PGD, 40 ), ( PGD, 100 ) ]
    eps_list = list(range(1, 41))
    eps_div = 100
    alpha_amp = 2.5/100

    in_channels = 1
    out_channels = 1
    depth = 3

class MEDICALconfig:
    method_and_iter = [ ( FGSM, 0 ), ( PGD, 7 ), ( PGD, 20 ) ]
    eps_list = list(range(1, 20))
    eps_div = 255
    alpha_amp = 1/4

    in_channels = 3
    out_channels = 3
    depth = 5