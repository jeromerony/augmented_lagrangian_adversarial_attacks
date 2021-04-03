import os
from functools import partial

import torch
from adv_lib.attacks import alma, ddn
from adv_lib.attacks.auto_pgd import minimal_apgd
from adv_lib.attacks.fast_adaptive_boundary import original_fab
from adv_lib.utils import requires_grad_
from adv_lib.utils.attack_utils import run_attack, compute_attack_metrics, print_metrics
from adv_lib.utils.lagrangian_penalties import all_penalties
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from attacks.foolbox import ead_attack
from models.mnist import SmallCNN, IBP_large

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
device = torch.device('cuda:0')
batch_size = 5000
os.makedirs(os.path.join('data', 'torchvision'), exist_ok=True)
os.makedirs(os.path.join('results', 'mnist'), exist_ok=True)

transform = transforms.ToTensor()
dataset = MNIST('data/torchvision', train=False, transform=transform, download=True)
loader = DataLoader(dataset=dataset, batch_size=10000)
images, labels = next(iter(loader))

models = {
    'SmallCNN_regular': SmallCNN(),
    'SmallCNN_ddn_l2': SmallCNN(),
    'SmallCNN_trades_linf': SmallCNN(),
    'IBP_large_linf': IBP_large(in_ch=1, in_dim=28),
}
models['SmallCNN_regular'].load_state_dict(torch.load('mnist_regular.pth'))
models['SmallCNN_ddn_l2'].load_state_dict(torch.load('mnist_robust_ddn.pth'))
models['SmallCNN_trades_linf'].load_state_dict(torch.load('mnist_robust_trades.pt'))
models['IBP_large_linf'].load_state_dict(torch.load('IBP_large_best.pth')['state_dict'])

[m.eval() for m in models.values()]
[m.to(device) for m in models.values()]
[requires_grad_(m, False) for m in models.values()]

penalty = all_penalties['P2']

attacks = [
    ('APGD_l2', partial(minimal_apgd, norm=2, targeted_version=True, max_eps=5, binary_search_steps=13)),

    ('EAD_l1_9x100', partial(ead_attack, steps=100)),
    ('EAD_l1_9x1000', partial(ead_attack, steps=1000)),

    ('FAB_l1_100', partial(original_fab, norm='L1', n_iter=100)),
    ('FAB_l1_1000', partial(original_fab, norm='L1', n_iter=1000)),
    ('FAB_l2_100', partial(original_fab, norm='L2', n_iter=100)),
    ('FAB_l2_1000', partial(original_fab, norm='L2', n_iter=1000)),

    ('DDN_100', partial(ddn, steps=100)),
    ('DDN_1000', partial(ddn, steps=1000)),

    ('ALMA_l1_100', partial(alma, penalty=penalty, distance='l1', init_lr_distance=0.5, α=0.5, num_steps=100)),
    ('ALMA_l1_1000', partial(alma, penalty=penalty, distance='l1', init_lr_distance=0.5, num_steps=1000)),
    ('ALMA_l2_100', partial(alma, penalty=penalty, distance='l2', init_lr_distance=0.1, α=0.5, num_steps=100)),
    ('ALMA_l2_1000', partial(alma, penalty=penalty, distance='l2', init_lr_distance=0.1, num_steps=1000)),
]

for name, method in attacks:
    torch.manual_seed(42)
    for model_name, model in models.items():
        print('\n{} - {}'.format(name, model_name))
        attack_data = run_attack(model=model, inputs=images, labels=labels, attack=method, batch_size=batch_size)
        attack_metrics = compute_attack_metrics(model=model, attack_data=attack_data)
        print_metrics(attack_metrics)
        torch.save(attack_metrics, 'results/mnist/metrics_{}_{}.pt'.format(model_name, name))
