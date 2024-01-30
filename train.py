import os
import time
import logging
import math
import argparse
import numpy as np
import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as data
import warnings
from util import setup_logger, print_args
from models import Trainer

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='KeDuSR Training')
    
    parser.add_argument('--dataset_root', default='./dataset/',type=str)
    parser.add_argument('--dataset_name', default='DuSR-Real', type=str) #Select a dataset [DuSR-Real, RealMCVSR-Real, CameraFusion-Real]
    parser.add_argument('--testloader', default='TestSet_cache', type=str) 
    parser.add_argument('--sr_scale', default=2, type=int)   # 2X super-resolution
    parser.add_argument('--random_seed', default=2023, type=int)
    parser.add_argument('--use_tb_logger', action='store_true')


    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--chunk_size', default=128, type=int) # Chunk size for inference
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--warm_up_iter', default=10000, type=int)
    parser.add_argument('--total_iter', default=250000, type=int)
    parser.add_argument('--log_freq', default=200, type=int)
    parser.add_argument('--test_freq', default=10000, type=int) 
    
    parser.add_argument('--loss_Charbonnier', action='store_true')
    parser.add_argument('--loss_perceptual', action='store_true')
    parser.add_argument('--loss_adv', action='store_true')

    parser.add_argument('--lambda_Charbonnier', default=1, type=float)
    parser.add_argument('--lambda_perceptual', default=1e-3, type=float)
    parser.add_argument('--lambda_adv', default=1e-4, type=float)
    
    parser.add_argument('--resume', default='', type=str)             #"weights/20240117_232231/snapshot/net_xxx.pth"
    parser.add_argument('--resume_optim', default='', type=str)       #"weights/20240117_232231/snapshot/optimizer_G_xxx.pth"
    parser.add_argument('--resume_scheduler', default='', type=str)   #"weights/20240117_232231/snapshot/scheduler_xxx.pth"

    parser.add_argument('--save_folder', default='./weights', type=str)

 
    args = parser.parse_args()
    set_random_seed(args.random_seed)


    if args.resume != '':
        args.save_folder =  os.path.join(args.save_folder, args.resume.split('/')[1])
    else:
        args.save_folder = os.path.join(args.save_folder, time.strftime('%Y%m%d_%H%M%S'))

    args.snapshot_save_dir = os.path.join(args.save_folder,  'snapshot')
    log_file_path = args.save_folder + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'


    if os.path.exists(args.snapshot_save_dir) == False:
        os.makedirs(args.snapshot_save_dir)

    setup_logger(log_file_path)
    print_args(args)

    cudnn.benchmark = True

    #train
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
