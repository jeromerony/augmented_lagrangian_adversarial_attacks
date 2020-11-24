import os

import pandas as pd
import torch

result_dir = 'results'

configs = {
    'mnist': {
        'models': [
            'SmallCNN_regular',
            'SmallCNN_ddn_l2',
            'SmallCNN_trades_linf',
            'IBP_large_linf',
        ],
        'distances': {
            'l1': [
                'EAD_l1_9x100',
                'EAD_l1_9x1000',
                'FAB_l1_100',
                'FAB_l1_1000',
                'ALMA_l1_100',
                'ALMA_l1_1000',
            ],
            'l2': [
                'DDN_100',
                'DDN_1000',
                'FAB_l2_100',
                'FAB_l2_1000',
                'APGD_l2',
                'ALMA_l2_100',
                'ALMA_l2_1000',
            ],

        }
    },
    'cifar10': {
        'models': [
            'WideResNet_28-10',
            'Carmon2019',
            'Augustin2020',
        ],
        'distances': {
            'l1': [
                'EAD_l1_9x100',
                'EAD_l1_9x1000',
                'FAB_l1_100',
                'FAB_l1_1000',
                'ALMA_l1_100',
                'ALMA_l1_1000',
            ],
            'l2': [
                'DDN_100',
                'DDN_1000',
                'FAB_l2_100',
                'FAB_l2_1000',
                'APGD_l2',
                'ALMA_l2_100',
                'ALMA_l2_1000',
            ],
            'ciede2000': [
                'Perc-AL_100',
                'Perc-AL_1000',
                'ALMA_CIEDE2000_100',
                'ALMA_CIEDE2000_1000',
                'ALMA_CIEDE2000_0.05_100',
                'ALMA_CIEDE2000_0.05_1000',
                'ALMA_CIEDE2000_0.1_100',
                'ALMA_CIEDE2000_0.1_1000',
            ],
            'ssim': [
                'ALMA_SSIM_100',
                'ALMA_SSIM_1000',
            ],
            'lpips': [
                'ALMA_LPIPS_100',
                'ALMA_LPIPS_1000',
            ]
        }
    },
    'imagenet': {
        'models': [
            'ResNet50',
            'ResNet50_l2_3',
            'ResNet50_linf_4',
        ],
        'distances': {
            'l1': [
                'EAD_l1_9x100',
                'EAD_l1_9x1000',
                'FAB_l1_100',
                'FAB_l1_1000',
                'ALMA_l1_100',
                'ALMA_l1_1000',
            ],
            'l2': [
                'DDN_100',
                'DDN_1000',
                'FAB_l2_100',
                'FAB_l2_1000',
                'APGD_l2',
                'ALMA_l2_100',
                'ALMA_l2_1000',
            ],
            'ciede2000': [
                'Perc-AL_100',
                'Perc-AL_1000',
                'ALMA_CIEDE2000_100',
                'ALMA_CIEDE2000_1000',
            ],
            'ssim': [
                'ALMA_SSIM_100',
                'ALMA_SSIM_1000',
            ],
            'lpips': [
                'ALMA_LPIPS_100',
                'ALMA_LPIPS_1000',
            ]
        }
    }
}

for dataset, config in configs.items():

    data = []
    for distance, attacks in config['distances'].items():
        for model in config['models']:
            for attack in attacks:
                metrics = torch.load(os.path.join(result_dir, dataset, 'metrics_{}_{}.pt'.format(model, attack)))

                adv_distances = metrics['distances'][distance]
                success = metrics['success']
                if 'SSIM' in distance:
                    adv_distances[~success] = -1
                else:
                    adv_distances[~success] = float('inf')

                data.append({
                    'distance': distance,
                    'model': model,
                    'attack': attack,
                    'ASR': success.float().mean().item(),
                    'median distance': torch.median(adv_distances).item(),
                    'num forwards': metrics.get('num_forwards', None),
                    'num backwards': metrics.get('num_backwards', None)
                })

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(result_dir, '{}.csv'.format(dataset)))