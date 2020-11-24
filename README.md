This repository contains the experiments for the paper "Augmented Lagrangian Adversarial Attacks". This **does not** contain the ALMA attack proposed in the paper, which is implemented in [adversarial-library](https://github.com/jeromerony/adversarial-library).


### Requirements
- python 3.8
- matplotlib>=3.3
- pandas>=1.1
- pytorch>=1.6
- torchvision>=0.7
- tqdm
- foolbox 3.2.1
- adversarial-library https://github.com/jeromerony/adversarial-library
- robustbench https://github.com/RobustBench/robustbench


### Additional required data

The model state dicts for MNIST, CIFAR10 and ImageNet are fetched from various locations. 

To ease reproducibility, we use the robustbench library to fetch the models for CIFAR10 (no action required here). For MNIST and ImageNet, the models can be fetched from their original repositories, however, we provide the models in a separate zip file to simplify the process. The zip file can downloaded at: https://drive.google.com/file/d/1PaoIhNV1PqPuYl2kG0_n7FdNCpHJuoFr/view?usp=sharing 

This zip file also contains the 1000 randomly selected images from the ImageNet validation set. These images have already been pre-processed (center-crop of 224x224) and stored into a pytorch Tensor.

Once downloaded, the files should be extracted at the root of this repository.

### Experiments

To run the experiments on MNIST, CIFAR10 and ImageNet, execute the scripts:
- `python minimal_attack_mnist.py`
- `python minimal_attack_cifar10.py`
- `python minimal_attack_imagenet.py`

These scripts assume that the code is run on the first visible cuda enabled device. Changing `torch.device('cuda:0')` to `torch.device('cpu')` allows to run them on CPU, however, this will be extremely slow. These scripts also assume that there is about 16GB of available video memory on the cuda device. For smaller memory sizes, `batch_size` can be reduced.

All the results will be saved in the `results` directory as `.pt` files containing python dictionaries with information related to the attacks. 

### Results

To extract all the results in a readable `.csv` file, use the `compile_results.py` script. This script contains a configuration of all the attacks run. If only a part of the experiments were performed, part of the config can be commented to account for it. This will create one `.csv` file per dataset and save them in the `results` directory.

### Curves

To plot the robust accuracy curves, the scripts `plot_results_mnist.py`, `plot_results_cifar10.py`, `plot_results_imagenet.py` can be executed. This will save the curves in the `results/curves` folder.