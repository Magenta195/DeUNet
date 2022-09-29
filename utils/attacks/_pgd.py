import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, device


class PGD():
    def __init__(
        self,
        device : device,
        loss : Module = nn.CrossEntropyLoss(),
        eps : float = 8/255,
        alpha : float = 2/255,
        iters : int = 7,
        **kwargs
    ) :
        self.loss = loss
        self.device = device
        self.eps = eps
        self.alpha = alpha
        self.iters = iters


    def __call__(
        self,
        model : Module,
        images : Tensor,
        labels : Tensor
    )  -> Tensor:

        images = images.to(self.device)
        labels = labels.to(self.device)
            
        ori_images = images.data
            
        for i in range(self.iters) :    
            images.requires_grad = True
            outputs = model(images)
            
            model.zero_grad()
            cost = self.loss(outputs, labels).to(self.device)
            cost.backward()

            adv_images = images + self.alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

        return images

    def get_name(self) :
        return self.__class__.__name__ + str( self.iters )