from typing import Optional

import torch
from PIL import Image
from adv_lib.utils import normalize_model, requires_grad_
from torchvision import transforms, models


def imagenet_model_factory(name: str, pretrained: bool = True, state_dict_path: Optional[str] = None, **kwargs):
    if 'inception' in name:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = [transforms.Resize(299, interpolation=Image.LANCZOS)]
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = [transforms.Resize(256, interpolation=Image.LANCZOS), transforms.CenterCrop(224)]

    model = models.__dict__[name](pretrained=pretrained, **kwargs)

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
            state_dict = {k[len('module.model.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    normalized_model = normalize_model(model=model, mean=mean, std=std)
    normalized_model.eval()
    requires_grad_(normalized_model, False)

    return normalized_model, transform
