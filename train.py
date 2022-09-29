
from torch import device

from utils import (
    get_arguments,
    get_dataset,
    train_one_epoch,
    fix_rand_seed,
    Recoder
)
from utils.attacks import PGD
from models import (
    Model,
    Model_With_Filter,
    Model_With_DUnet_Filter
)

if __name__ == '__main__' :
    args = get_arguments()
    fix_rand_seed(args.seed)

    trainloader, testloader = get_dataset(args)
    dev = device( f'cuda:{args.device}' )

    model = None
    attack = None
    if not args.adv_train :
        model = Model(args).to(dev)
    else :
        attack = PGD(
                device = dev,
                eps = args.train_eps,
                alpha = args.train_alpha,
                iters = args.train_iter
                )
        if args.filter :
            model = Model_With_Filter(args).to(dev)
            model.model_load( isbase = True )

        elif args.HGD :
            model = Model_With_DUnet_Filter(args).to(dev)
            model.model_load( isbase = True )       
        
        else :
            model = Model(args).to(dev)

    recoder = Recoder(
        args = args,
        model = model,
        is_training = True,
        scope = 'maxacc' if not args.adv_train else 'minloss'
    )

    for epoch in range( args.epochs ):
        train_acc, train_loss, val_acc, val_loss, train_time = train_one_epoch(
            cur_epoch = epoch,
            total_epoch = args.epochs,
            attack = attack,
            model = model,
            trainloader = trainloader,
            testloader = testloader,
            dev = dev
        )

        recoder.update(
            train_acc = train_acc,
            train_loss = train_loss,
            train_time = train_time,
            val_acc = val_acc,
            val_loss = val_loss,
            epoch = epoch
        )