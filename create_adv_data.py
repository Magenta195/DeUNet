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
import torch
import numpy as np

DATASET_RESULT_PATH = './data/synthetic_medical'
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
    for i in range(args.cls) :
        os.makedirs(os.path.join(TRAIN_RESULT_PATH, str(i)), exist_ok=True)
        os.makedirs(os.path.join(TEST_RESULT_PATH, str(i)), exist_ok=True)

    for method, iters, eps in [(FGSM, 1, 16/255), (PGD, 20, 1/255), (PGD, 20, 8/255), (PGD, 20, 16/255)] :
        alpha = eps / 4
        attack = method(
            device = dev,
            eps = eps,
            alpha = alpha,
            iter = iters
        )
        
        for dset_result_path, dataloader in [ (TRAIN_RESULT_PATH, trainloader), (TEST_RESULT_PATH, testloader) ]:

            for i, batch in enumerate( tqdm( dataloader, leave=True) ):
                inputs: Tensor = batch[0].to(dev)
                labels: Tensor = batch[1].to(dev)

                orig_inputs = inputs.data
                inputs = attack( model.base_model, inputs,  labels )
                
                for orig_input, input, label in zip(orig_inputs, inputs, labels) :
                    _final_input = torch.cat((orig_input, input), 1)
                    DATA_PATH = os.path.join(dset_result_path, str(label.item()), str(dset_idx[label.item()]))
                    final_input = _final_input.cpu().numpy()
                    np.save(DATA_PATH, final_input) 
                    dset_idx[label.item()] += 1

    print(dset_idx)

