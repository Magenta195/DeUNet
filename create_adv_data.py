from torch import device, Tensor
from torchvision.utils import save_image
from utils import (
    get_arguments,
    get_dataset,
    get_cfg,
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

from collections import defaultdict
from tqdm import tqdm
import os 


DATASET_RESULT_PATH = './data/syntetic_medical'
TRAIN_RESULT_PATH = os.path.join(DATASET_RESULT_PATH, 'train')
TEST_RESULT_PATH = os.path.join(DATASET_RESULT_PATH, 'test')

if __name__ == '__main__' :
    args = get_arguments()
    fix_rand_seed(args.seed)

    trainloader, testloader = get_dataset(args)
    dev = device( f'cuda:{args.device}' )

    model = Model(args).to(dev)

    dset_idx = defaultdict(int)
    os.makedirs(DATASET_RESULT_PATH, exist_ok=True)
    os.makedirs(TRAIN_RESULT_PATH, exist_ok=True)
    os.makedirs(TEST_RESULT_PATH, exist_ok=True)
    for i in range(1, args.cls+1) :
        os.makedirs(os.path.join(TRAIN_RESULT_PATH, str(i)), exist_ok=True)
        os.makedirs(os.path.join(TEST_RESULT_PATH, str(i)), exist_ok=True)

    for method, iters, eps in [(FGSM, 1, 16/255), (PGD, 20, 1/255), (PGD, 20, 16/255)] :
        alpha = eps / 4
        attack = method(
            device = dev,
            eps = eps,
            alpha = alpha,
            iter = iters
        )
        
        for dset_name, dataloader in [ (TRAIN_RESULT_PATH, trainloader), (TEST_RESULT_PATH, testloader) ]:

            for i, batch in enumerate( tqdm( dataloader, leave=True) ):
                inputs: Tensor = batch[0].to(dev)
                labels: Tensor = batch[1].to(dev)

                orig_inputs = inputs.data
                inputs = attack( model.base_model, inputs,  labels )
                
                for input, label in zip(inputs, labels) :
                    DATA_PATH = os.path.join(dset_name, dset_idx[label.item])
                    save_image(input, DATA_PATH)   
                    dset_idx[label.item] += 1

    print(dset_idx)

