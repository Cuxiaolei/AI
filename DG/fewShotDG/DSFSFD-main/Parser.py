# -*- coding:utf-8 -*-

import argparse

def Parser():
    parser = argparse.ArgumentParser(description='Cross domain few shot learning parameters')
    parser.add_argument('-dp', '--data_dir', type=str, help='Dataset root in our experiment', default='D:/KAT_DOMAIN')
    parser.add_argument('-bp', '--base_dir', type=str, help='Base root in our experiment', default='D:/KAT_test0117')
    parser.add_argument('-tdom', '--train_domain', type=str, help='the domain that we used to train the model', default='N09_M07_F10')
    parser.add_argument('-tedom', '--test_domain', type=str, help='the domain that we used to test the model', default='N15_M01_F10')
    parser.add_argument('-kt', '--k_train', type=int, help='number of classes during training', default=5)
    parser.add_argument('-nt', '--n_train', type=int, help='number of support sample in each class during training', default=5)
    parser.add_argument('-qt', '--q_train', type=int, help='number of query sampler in each class during training', default=15)
    parser.add_argument('-kv', '--k_val', type=int, help='number of classing during val and test', default=5)
    parser.add_argument('-nv', '--n_val', type=int, help='number of support sample in each class during val and test', default=5)
    parser.add_argument('-qv', '--q_val', type=int, help='number of query sampler in each class during val and test', default=15)
    parser.add_argument('-ep', '--epochs', type=int, help='number of epoch in model train', default=50)
    parser.add_argument('-epi', '--episodes', type=int, help='number of episodes in each epoch', default=100)
    parser.add_argument('-mlr', '--lr', type=float, help='the learning rate during optimize the model except FT', default=1e-3)
    parser.add_argument('-ftlr', '--ft_lr', type=float, help='the learning rate during optimize FT layer', default=1e-3)
    parser.add_argument('-DEW', '--DE_weight', type=float, help='the contribution of Time domain for predition', default=0.6)
    parser.add_argument('-FFTW', '--FFT_weight', type=float, help='the contribution of Frequency domain for predition', default=0.15)
    parser.add_argument('-TFW', '--TF_weight', type=float, help='the contribution of Time Frequency domain for predition', default=0.25)

    return parser