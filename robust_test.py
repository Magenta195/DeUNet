
from torch import device

from utils import (
    get_arguments,
    get_dataset,
    robust_test,
    fix_rand_seed,
    Recoder
)
from utils.attacks import PGD, FGSM
from models import (
    Model,
    Model_With_Filter,
    Model_With_DUnet_Filter
)

eps_list = list( x / 255 for x in range(1, 20))
iter_list = [ 7, 20 ]

if __name__ == '__main__' :
    args = get_arguments()
    fix_rand_seed(args.seed)

    _, testloader = get_dataset(args)
    dev = device( f'cuda:{args.device}' )

    if args.filter :
        model = Model_With_Filter(args).to(dev)
        model.model_load( isbase = True )
        model.model_load( isbase = False )
    elif args.HGD :
        model = Model_With_DUnet_Filter(args).to(dev)
        model.model_load( isbase = True )
        model.model_load( isbase = False )        
    else :
        model = Model(args).to(dev)
        model.model_load( isbase = True )

    recoder = Recoder(
        args = args,
        model = model,
        is_training = False,
    )

    eps_list = list( x for x in range(1, 20))
    for eps in eps_list :        
        for method, iters in [ ( FGSM, 0), (PGD, 7), (PGD, 20) ] :    
            attack = None
            attack_method = 'None'
            
            _eps = eps / 255

            if method is not None :
                attack = method(        
                        device = dev,
                        eps = _eps,
                        alpha = _eps / 4,
                        iters = iters
                )
                attack_method = attack.get_name()

            val_attack_success, val_acc, val_time = robust_test(
                model = model,
                attack = attack,
                dataloader = testloader,
                dev = dev
            )

            recoder.update(
                method = attack_method,
                epsilon = eps,
                accuracy = val_acc,
                success_rate = val_attack_success,
                val_time = val_time
            )
            


