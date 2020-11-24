from foolbox.attacks import EADAttack
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.models import PyTorchModel
from torch import nn, Tensor


def ead_attack(model: nn.Module,
               inputs: Tensor,
               labels: Tensor,
               targeted: bool = False,
               **kwargs) -> Tensor:
    fmodel = PyTorchModel(model=model, bounds=(0, 1))
    attack = EADAttack(**kwargs)
    if targeted:
        criterion = TargetedMisclassification(target_classes=labels),
    else:
        criterion = Misclassification(labels=labels)
    adv_inputs = attack(model=fmodel, inputs=inputs, criterion=criterion, epsilons=None)[0]

    return adv_inputs
