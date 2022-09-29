import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, device


class FGSM():
    def __init__(
        self,
        device : device,
        loss : Module = nn.CrossEntropyLoss(),
        eps : float = 8/255,
        **kwargs
    ) :
        self.loss = loss
        self.device = device
        self.eps = eps


    def __call__(
        self,
        model : Module,
        images : Tensor,
        labels : Tensor
    )  -> Tensor:

        images = images.to(self.device)
        labels = labels.to(self.device)
            
        images.requires_grad = True
        outputs = model(images)
            
        model.zero_grad()
        cost = self.loss(outputs, labels).to(self.device)
        cost.backward()

        adv_images = images + self.eps * images.grad.sign()
        images = torch.clamp(adv_images, min=0, max=1).detach_()

        return images

    def get_name(self) :
        return self.__class__.__name__