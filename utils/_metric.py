from typing import Optional, Tuple
from torch import Tensor
import torch

def accuracy(
    prediction: Tensor,
    label: Tensor,
    ) -> Tensor :
    
    accuracy_size = torch.sum( prediction.data == label.data )
    return accuracy_size

def attack_success_rate(
    prediction: Tensor,
    label: Tensor,
    attacked_prediction: Tensor,
    filtered_prediction: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor] :

    success_prediction = prediction == label.data
    success_size = torch.sum(success_prediction)
    attack_success = attacked_prediction != label.data
    if filtered_prediction is None :
        attack_success = attacked_prediction != label.data
        attack_success_size = torch.sum( attack_success * success_prediction )

    else :
        attack_detected = attacked_prediction != filtered_prediction
        attack_success_not_detected = attack_success * (~ attack_detected )
        attack_success_size = torch.sum(attack_success_not_detected * success_prediction)

    return success_size, attack_success_size

if __name__ == '__main__' :
    orig_pred =     torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    orig_label =    torch.Tensor([0, 0, 2, 2, 4, 5, 6, 7, 8, 9])

    attacked_pred = torch.Tensor([1, 0, 2, 4, 3, 6, 5, 7, 8, 9])
    filtered_pred = torch.Tensor([0, 0, 2, 3, 4, 5 ,6, 7, 8, 9])
    print(accuracy(orig_pred, orig_label))
    print(accuracy(attacked_pred, orig_label))
    print(attack_success_rate(
        prediction = orig_pred,
        label = orig_label,
        attacked_prediction = attacked_pred,
        filtered_prediction = filtered_pred
    ))