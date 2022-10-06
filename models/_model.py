import torch.nn as nn
import torch.optim as optim
import torch
import os

from torch import Tensor

from ._resnet import resnet_models
from ._simple_model import CNN
from ._denosing_filter import unet_denoising, dunet
from utils import get_cfg

def get_base_models(
    model_name : str,
    num_cls : int,
) -> nn.Module :
    if 'resnet' in model_name :
        return resnet_models(model_name, num_cls)
    else :
        return CNN(num_cls)

class Model(nn.Module) :
    def __init__(self, args) :
        super().__init__()
        self.cfg = get_cfg(args)
        self.base_model = get_base_models(args.model, args.cls)
        self.save_path = args.save_path
        self.base_load_path = args.base_load_path

        self.fname = args.fname
        self.isfiltered = False
        self.device = torch.device( f'cuda:{args.device}' )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.base_model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.0001
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=[100, 150],
            gamma=0.1
        )

    def cal_loss(self, orig_inputs, inputs, labels) :
        out = self.forward(inputs)
        return out, self.criterion(out, labels)

    def forward(self, x):
        out = self.base_model(x)
        return out

    def model_save(self, **kwarg):
        SAVE_PATH = os.path.join(self.save_path, self.fname)

        os.makedirs(self.save_path, exist_ok = True)
        torch.save(self.base_model.state_dict(), SAVE_PATH)

    def model_load(self, **kwarg):
        LOAD_PATH = os.path.join(self.base_load_path, self.fname)
        self.base_model.load_state_dict(torch.load(LOAD_PATH, map_location=self.device))

    def is_filtered(self) :
        return self.isfiltered

class Model_With_Filter(Model):
    def __init__(self, args):
        super().__init__(args)
        self.filter_model = unet_denoising(
            in_channels = self.cfg.in_channels,
            out_channels = self.cfg.out_channels,
            depth = self.cfg.depth
        )
        self.load_path = args.load_path

        self.criterion = _CE_simil_loss(beta = args.beta)
        self.optimizer = optim.SGD(
            self.filter_model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.0001
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=[100, 150],
            gamma=0.1
        )
        self.isfiltered = True

    def cal_loss(self, orig_inputs, inputs, labels) :
        filtered_inputs = inputs - self.filter_model(inputs)
        filtered_feature_out = self.base_model.feature_forward( filtered_inputs )
        feature_out = self.base_model.feature_forward( orig_inputs )
        filtered_out = self.base_model.classifier_forward( filtered_feature_out )

        loss = self.criterion(
                inputs = orig_inputs,
                filtered_inputs = filtered_inputs,
                filtered_outputs = filtered_feature_out,
                target = feature_out,
            )

        return filtered_out, loss

    def forward(self, x):
        filtered_x = x - self.filter_model(x)
        out = self.base_model(filtered_x)
        
        return out

    def model_save(self, isbase) :
        if isbase :
            super().model_save()

        else :
            SAVE_PATH = os.path.join(self.save_path, self.fname)

            os.makedirs(self.save_path, exist_ok = True)
            torch.save(self.filter_model.state_dict(), SAVE_PATH)


    def model_load(self, isbase):
        if isbase :
            super().model_load()

        else :
            LOAD_PATH = os.path.join(self.load_path, self.fname)
            self.filter_model.load_state_dict(torch.load(LOAD_PATH, map_location=self.device))

    def is_filtered(self):
        return self.isfiltered

class Model_With_DUnet_Filter(Model):
    def __init__(self, args):
        super().__init__(args)
        self.filter_model = dunet(
            in_channels = self.cfg.in_channels,
            out_channels = self.cfg.out_channels,
            depth = self.cfg.depth
        )
        self.load_path = args.load_path
        self.criterion = nn.L1Loss(reduction = 'mean')
        self.optimizer = optim.SGD(
            self.filter_model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.0001
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=[100, 150],
            gamma=0.1
        )
        self.isfiltered = True

    def cal_loss(self, orig_inputs, inputs, labels) : # FGD
        filtered_inputs = inputs - self.filter_model(inputs)
        filtered_feature_out = self.base_model.feature_forward( filtered_inputs )
        feature_out = self.base_model.feature_forward( orig_inputs )
        filtered_out = self.base_model.classifier_forward( filtered_feature_out )

        loss = self.criterion(feature_out, filtered_feature_out)
        
        return filtered_out, loss

    def forward(self, x):
        filtered_x = x - self.filter_model(x)
        out = super().forward(filtered_x)

        return out

    def model_save(self, isbase) :
        if isbase :
            super().model_save()

        else :
            SAVE_PATH = os.path.join(self.save_path, self.fname)

            os.makedirs(self.save_path, exist_ok = True)
            torch.save(self.filter_model.state_dict(), SAVE_PATH)


    def model_load(self, isbase):
        if isbase :
            super().model_load()
            
        else :
            LOAD_PATH = os.path.join(self.load_path, self.fname)
            self.filter_model.load_state_dict(torch.load(LOAD_PATH, map_location=self.device))

    def is_filtered(self):
        return self.isfiltered

class _CE_simil_loss(nn.CrossEntropyLoss):
    def __init__(
        self,
        reduction : str = 'mean',
        beta : float = 1.,
        **kwargs,
    ):
        super().__init__(reduction = 'none', **kwargs)
        self.l1loss = nn.L1Loss(reduction = 'mean')
        self.beta = beta
        self.reduction = reduction

    def forward(self,
        inputs : Tensor,
        filtered_inputs : Tensor,
        filtered_outputs : Tensor,
        target : Tensor,
    ) -> Tensor:

        # ce_loss = super().forward(filtered_outputs, target)
        # l1_loss = self.l1loss(inputs, filtered_inputs)
        # total_loss = self.beta * l1_loss + ce_loss
        l1_feature_loss = self.l1loss(filtered_outputs, target)
        total_loss = l1_feature_loss
        # total_loss = l1_loss + l1_feature_loss

        if self.reduction == 'mean' :
            return torch.mean(total_loss)
        else :
            return total_loss