import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    # Dataset and Dataloader
    parser.add_argument( '-b', '--batch', type=int, default=128 )
    parser.add_argument( '-i', '--image', type=int, default=32 )
    parser.add_argument( '-c', '--cls', type=int, default=10)
    parser.add_argument( '--dataset', type=str, default='cifar10' )
    parser.add_argument( '--dataset_path', type=str, default='./data/cifar10' )

    # Optimizer and Scheduler
    parser.add_argument( '--lr', type=float, default=1e-3 )

    # Random seed
    parser.add_argument( '-s', '--seed', type=int, default=42 )

    # epoch
    parser.add_argument( '-e', '--epochs', type=int, default=200 )

    # mode
    parser.add_argument( '--wandb', action='store_true', default=False )

    # model
    parser.add_argument( '--model', type=str, default='resnet-50')
    parser.add_argument( '--load_path', type=str, default='./result')
    parser.add_argument( '--base_load_path', type=str, default='./result')
    parser.add_argument( '--save_path', type=str, default='./result')
    parser.add_argument( '--fname', type=str, default='best_model.pth')
    parser.add_argument( '--device', type=int, default=0)

    # for training
    parser.add_argument( '--train', default=True, action='store_false')
    parser.add_argument( '--adv_train', default=False, action='store_true')
    parser.add_argument( '--filter', default=False, action='store_true' )
    parser.add_argument( '--HGD', default=False, action='store_true')
    parser.add_argument( '--train_iter', type=int, default=20)
    parser.add_argument( '--train_eps', type=float, default=8/255)
    parser.add_argument( '--train_alpha', type=float, default=2/255)
    parser.add_argument( '--beta', type=float, default=1.)

    # for testing


    return show_arguments( parser.parse_args() )


def show_arguments(args):
    var = vars(args)
    var_keys = list( var.keys() )

    print("[ Adversarial denosing ] Arguments setup ")
    for name in var_keys:
        print(f"{name} >>> {var[name]}")
    print()

    return args