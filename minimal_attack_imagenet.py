import os
from functools import partial

import torch
from adv_lib.attacks import alma, ddn
from adv_lib.attacks.auto_pgd import minimal_apgd
from adv_lib.attacks.perceptual_color_attacks import perc_al
from adv_lib.distances.color_difference import ciede2000_loss
from adv_lib.distances.lpips import LPIPS
from adv_lib.distances.structural_similarity import compute_ssim, compute_ms_ssim
from adv_lib.utils.attack_utils import run_attack, compute_attack_metrics, print_metrics, _default_metrics
from adv_lib.utils.lagrangian_penalties import all_penalties

from attacks.foolbox import ead_attack
from attacks.original_fab import original_fab
from models.imagenet import imagenet_model_factory

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
device = torch.device('cuda:0')
batch_size = 128
os.makedirs(os.path.join('results', 'imagenet'), exist_ok=True)

models = {
    'ResNet50': imagenet_model_factory('resnet50')[0],
    'ResNet50_l2_3': imagenet_model_factory('resnet50', state_dict_path='imagenet_l2_3_0.pt')[0],
    'ResNet50_linf_4': imagenet_model_factory('resnet50', state_dict_path='imagenet_linf_4.pt')[0],
}
[m.to(device) for m in models.values()]

data = torch.load('imagenet_1000_random.pth')
images, labels = data['images'], data['labels']
images = images.float() / 255

penalty = all_penalties['P2']

attacks = [
    ('APGD_l2', partial(minimal_apgd, norm=2, targeted_version=True, max_eps=10, binary_search_steps=14)),

    ('EAD_l1_9x100', partial(ead_attack, steps=100)),
    ('EAD_l1_9x1000', partial(ead_attack, steps=1000)),

    ('FAB_l1_100', partial(original_fab, norm='L1', n_iter=100, targeted_variant=True)),
    ('FAB_l1_1000', partial(original_fab, norm='L1', n_iter=1000, targeted_variant=True)),
    ('FAB_l2_100', partial(original_fab, norm='L2', n_iter=100, targeted_variant=True)),
    ('FAB_l2_1000', partial(original_fab, norm='L2', n_iter=1000, targeted_variant=True)),

    ('DDN_100', partial(ddn, steps=100)),
    ('DDN_1000', partial(ddn, steps=1000)),

    ('Perc-AL_100', partial(perc_al, num_classes=10, max_iterations=100)),
    ('Perc-AL_1000', partial(perc_al, num_classes=10, max_iterations=1000)),

    ('ALMA_l1_100', partial(alma, penalty=penalty, distance='l1', init_lr_distance=0.5, α=0.5, num_steps=100)),
    ('ALMA_l1_1000', partial(alma, penalty=penalty, distance='l1', init_lr_distance=0.5, num_steps=1000)),
    ('ALMA_l2_100', partial(alma, penalty=penalty, distance='l2', init_lr_distance=0.1, α=0.5, num_steps=100)),
    ('ALMA_l2_1000', partial(alma, penalty=penalty, distance='l2', init_lr_distance=0.1, num_steps=1000)),
    ('ALMA_SSIM_100', partial(alma, penalty=penalty, distance='ssim', init_lr_distance=0.0001, α=0.5, num_steps=100)),
    ('ALMA_SSIM_1000', partial(alma, penalty=penalty, distance='ssim', init_lr_distance=0.0001, num_steps=1000)),
    ('ALMA_CIEDE2000_100', partial(alma, penalty=penalty, distance='ciede2000', init_lr_distance=0.03, α=0.5, num_steps=100)),
    ('ALMA_CIEDE2000_1000', partial(alma, penalty=penalty, distance='ciede2000', init_lr_distance=0.03, num_steps=1000)),
    ('ALMA_LPIPS_100', partial(alma, penalty=penalty, distance='lpips', init_lr_distance=0.01, α=0.5, num_steps=100)),
    ('ALMA_LPIPS_1000', partial(alma, penalty=penalty, distance='lpips', init_lr_distance=0.01, num_steps=1000)),
]

metrics = _default_metrics
metrics['ssim'] = compute_ssim
metrics['msssim'] = compute_ms_ssim
metrics['ciede2000'] = ciede2000_loss
metrics['lpips'] = partial(LPIPS, linear_mapping='alex.pth')

for name, method in attacks:
    torch.manual_seed(42)
    for model_name, model in models.items():
        print('\n{} - {}'.format(name, model_name))
        attack_data = run_attack(model=model, inputs=images, labels=labels, attack=method, batch_size=batch_size)
        attack_metrics = compute_attack_metrics(model=model, attack_data=attack_data)
        print_metrics(attack_metrics)
        torch.save(attack_metrics, 'results/imagenet/metrics_{}_{}.pt'.format(model_name, name))
