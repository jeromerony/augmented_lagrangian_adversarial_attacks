import os

import matplotlib.pyplot as plt
import torch

from utils import robust_accuracy_curve

result_dir = os.path.join('results', 'imagenet')
os.makedirs(os.path.join('results', 'curves'), exist_ok=True)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['lines.linewidth'] = 1
fontsize = 8
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['axes.titlesize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

configs = {
    'l1': {
        'norm': '$\ell_1$-norm',
        'models': [
            ('ResNet50', 200),
            ('ResNet50_l2_3', 1000),
            ('ResNet50_linf_4', 300),
        ],
        'attacks': [
            ('EAD_l1_9x100', r'EAD 9$\times$100', ':', colors[0]),
            ('EAD_l1_9x1000', r'EAD 9$\times$1000', '-', colors[0]),
            ('FAB_l1_100', '$\mathrm{FAB}^\mathrm{T}$ $\ell_1$ 100', ':', colors[1]),
            ('FAB_l1_1000', '$\mathrm{FAB}^\mathrm{T}$ $\ell_1$ 1000', '-', colors[1]),
            ('ALMA_l1_100', 'ALMA 100', ':', colors[2]),
            ('ALMA_l1_1000', 'ALMA 1000', '-', colors[2]),
        ]
    },
    'l2': {
        'norm': '$\ell_2$-norm',
        'models': [
            ('ResNet50', 1),
            ('ResNet50_l2_3', 15),
            ('ResNet50_linf_4', 8),
        ],
        'attacks': [
            ('DDN_100', 'DDN 100', ':', colors[0]),
            ('DDN_1000', 'DDN 1000', '-', colors[0]),
            ('FAB_l2_100', '$\mathrm{FAB}^\mathrm{T}$ $\ell_2$ 100', ':', colors[1]),
            ('FAB_l2_1000', '$\mathrm{FAB}^\mathrm{T}$ $\ell_2$ 1000', '-', colors[1]),
            ('APGD_l2', '$\mathrm{APGD}^\mathrm{T}_\mathrm{DLR}$ $\ell_2$', '-', colors[3]),
            ('ALMA_l2_100', 'ALMA 100', ':', colors[2]),
            ('ALMA_l2_1000', 'ALMA 1000', '-', colors[2]),
        ]
    },
    'ciede2000': {
        'norm': 'Accumulated CIEDE2000',
        'models': [
            ('ResNet50', 6),
            ('ResNet50_l2_3', 50),
            ('ResNet50_linf_4', 50),
        ],
        'attacks': [
            ('Perc-AL_100', 'PerC-AL 100', ':', colors[0]),
            ('Perc-AL_1000', 'PerC-AL 1000', '-', colors[0]),
            ('ALMA_CIEDE2000_100', 'ALMA CIEDE2000 100', ':', colors[2]),
            ('ALMA_CIEDE2000_1000', 'ALMA CIEDE2000 1000', '-', colors[2]),
        ]
    }
}

for distance, config in configs.items():

    for model, xlim_high in config['models']:
        fig, ax = plt.subplots(figsize=(3.2, 2.4))

        for attack, legend, linestyle, color in config['attacks']:
            metrics = torch.load(os.path.join(result_dir, 'metrics_{}_{}.pt'.format(model, attack)))

            adv_distances = metrics['distances'][distance]
            success = metrics['success']

            distances, robust_accuracies = robust_accuracy_curve(distances=adv_distances, successes=success)
            ax.plot(distances, robust_accuracies, label=legend, linestyle=linestyle, c=color)

        ax.legend()
        yticks = [0, 0.25, 0.5, 0.75, 1]
        yticklabels = [0, 25, 50, 75, 100]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlim(0, xlim_high)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Robust Accuracy (\%)')
        ax.set_xlabel(config['norm'])
        plt.grid(True, linestyle='--', c='lightgray', which='both')
        fig.savefig('results/curves/attack_curves_imagenet_{}_{}.pdf'.format(distance, model), bbox_inches='tight')

configs = {
    'l1': {
        'norm': '$\ell_1$-norm',
        'models': [
            ('ResNet50', 'ResNet-50', 200),
            ('ResNet50_l2_3', 'ResNet-50 $\ell_2$ adv. trained ($\epsilon=3.0$)', 1000),
        ],
        'attacks': [
            ('EAD_l1_9x100', r'EAD 9$\times$100', ':', colors[0]),
            ('EAD_l1_9x1000', r'EAD 9$\times$1000', '-', colors[0]),
            ('FAB_l1_100', 'FAB $\ell_1$ 100', ':', colors[1]),
            ('FAB_l1_1000', 'FAB $\ell_1$ 1000', '-', colors[1]),
            ('ALMA_l1_100', 'ALMA 100', ':', colors[2]),
            ('ALMA_l1_1000', 'ALMA 1000', '-', colors[2]),
        ]
    },
}

for distance, config in configs.items():

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.))
    for i, (ax, (model, model_name, xlim_high)) in enumerate(zip(axes, config['models'])):

        for attack, legend, linestyle, color in config['attacks']:
            metrics = torch.load(os.path.join(result_dir, 'metrics_{}_{}.pt'.format(model, attack)))

            adv_distances = metrics['distances'][distance]
            success = metrics['success']

            distances, robust_accuracies = robust_accuracy_curve(distances=adv_distances, successes=success)
            ax.plot(distances, robust_accuracies, label=legend, linestyle=linestyle, c=color)

        yticks = [0, 0.1, 0.25, 0.5, 0.75, 1]
        yticklabels = [0, 10, 25, 50, 75, 100]
        ax.set_title(model_name, pad=3)
        ax.set_yticks(yticks)
        ax.set_xlim(0, xlim_high)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', pad=2)
        if i == 0:
            ax.legend()
            ax.set_yticklabels(yticklabels)
            ax.set_ylabel('Robust Accuracy (\%)', labelpad=2)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel(config['norm'], labelpad=2)
        ax.grid(True, linestyle='--', c='lightgray', which='both')

    fig.subplots_adjust(left=0.09, bottom=0.15, right=0.97, top=0.92, wspace=0.08, hspace=0)
    fig.savefig('results/curves/attack_curves_imagenet_{}_combined.pdf'.format(distance))
