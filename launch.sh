torchrun --nproc_per_node=1 train.py --depth=16 --bs=256 --ep=150 --patch_size=8 --pn=1_2_3_4 --data_path=datasets/cifar10 --exp_name=better
