from torch.nn.modules import Module
from torch.optim import Optimizer

import wandb


env_config = {
    "Name": None,
    "Random_seed": None,
    "Dataset": {
        "name": None,
        "batch size": None,
        "image size": None,
    },
    "Optimizer": {
        "name": None,
        "learning rate": None,
        "momentum": None,
        "weight decay": None,
    },
    "Scheduler": {
        "name": None,
        "Info": None,
    },
    "filtered": None,
    "adv_trained": None,
    "adv_attack " : {
        "name": None,
        "eps": None,
        "alpha": None,
        "iter": None,
    },
}


def wandb_config( 
    args,
    criterion: Module,
    optimizer: Optimizer, 
    is_training: bool,
    scheduler = None,
    wandb_project_name: str = 'denoising_adv',
    wandb_entity_name: str = 'Magenta195',
) -> wandb:

    batch_size = args.batch


    name=f'{args.model}'
    
    if args.adv_train :
        name += ' / adv_train'
    if args.filter :
        name += ' / filtered'
    if args.HGD :
        name += ' / HGD'
    if not is_training :
        name += ' / validation'

    name_optim = str(optimizer.__class__)
    name_optim = name_optim.split(".")
    name_optim = name_optim[-1].replace("'>", "")

    name_sched = None
    if scheduler != None:
        name_sched = str(scheduler.__class__)
        name_sched = name_sched.split(".")
        name_sched = name_sched[-1].replace("'>", "")

    tags = []
    tags.append( f'{args.model}')
    tags.append( f'image size {args.image}' )
    tags.append( f'batch size {batch_size}')
    tags.append( f'{name_optim}')
    tags.append( f'{name_sched}')

    env_config["Name"] = name
    env_config["Random_seed"] = args.seed
    env_config["Criterion"] = criterion.reduction

    env_config["Dataset"]["name"] = args.dataset
    env_config["Dataset"]["batch_size"] = batch_size
    env_config["Dataset"]["image_size"] = args.image

    env_config["filtered"] = args.filter
    env_config["adv_trained"] = args.adv_train

    if is_training :
        env_config["Optimizer"]["name"] = f"{name_optim}"
        env_config["Optimizer"]["learning_rate"] = optimizer.param_groups[0]['lr']
        env_config["Optimizer"]["weight_decay"] = optimizer.param_groups[0]['weight_decay']
        try:
            env_config["Optimizer"]["momentum"] = optimizer.param_groups[0]['momentum']
        except:
            env_config["Optimizer"]["momentum"] = None

        env_config["Scheduler"]["name"] = f"{name_sched}"
        env_config["Scheduler"]["Info"] = f"Start LR: {optimizer.param_groups[0]['lr']}"

    ## Create Wandb
    record_wandb = wandb
    record_wandb.init(
        project=wandb_project_name,
        entity=wandb_entity_name,
        name=name,
        tags=tags,
        config=env_config,
    )

    return record_wandb
