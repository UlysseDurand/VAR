This is a fork of https://github.com/FoundationVision/VAR

We tried to propose a new model using VQWGAN instead of VQGAN in the tokenization part, and we will compare the results to the basic model.

The experiments on this branch highly rely on the code from the original VAR repository. The new files are:
- VAR/models/vqwgan.py: Implementation of the VQWGAN model.
- VAR/train_var_vqwgan_cifar10.py: Training script for VAR with VQWGAN on CIFAR-10 dataset.
- VAR/train_vqwgan_cifar10.py: Training script for standalone VQWGAN on CIFAR-10 dataset.
- VAR/gen_var_class.py : Sample generation script using the trained VAR with VQWGAN model.

Dziki Yanis/ Ulysse Durand