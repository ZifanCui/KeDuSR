import os
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from util import setup_logger, print_args
import numpy as np
import warnings
import logging
import importlib
import json
from torch.utils.data import DataLoader
from models.archs.KeDuSR_arch import KeDuSR
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from crop_psnr import calculate_ssim, calculate_psnr
import torchvision
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict

def prepare(batch_samples):
    for key in batch_samples.keys():
        if 'name' not in key:
            batch_samples[key] = batch_samples[key].to('cuda')
    return batch_samples


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='KeDuSR Inference')

    parser.add_argument('--dataset_root', default='./dataset',type=str) 
    parser.add_argument('--dataset_name', default='DuSR-Real', type=str)
    parser.add_argument('--resume', default='./pre-trained/DuSR-Real_220000.pth', type=str)
    parser.add_argument('--result_dir', default='./inference_result', type=str)
    parser.add_argument('--chunk_size', default=128, type=int)
    parser.add_argument('--testloader', default='TestSet_cache', type=str)
    parser.add_argument('--sr_scale', default=2, type=int)
    args = parser.parse_args()


    save_dir_name = os.path.basename(args.resume)[:-4]
    args.save_dir = os.path.join(args.result_dir, args.dataset_name, save_dir_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_file_path = os.path.dirname(args.save_dir) + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'
    setup_logger(log_file_path)
    print_args(args)
    cudnn.benchmark = True
    logging.info('{}'.format(args.resume))


    # init test dataloader
    testset = getattr(importlib.import_module('dataloader.dataset'), args.testloader, None)
    test_dataset = testset(args)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)

    with open('./corrdinate/corrdinate_{}.json'.format(args.dataset_name),'r') as f:
        corrdinates = json.load(f)


    net = KeDuSR(args).cuda()

    if args.resume:
        if isinstance(net, nn.DataParallel) or isinstance(net, DistributedDataParallel): 
            net = net.module     
        load_net = torch.load(args.resume, map_location=torch.device('cuda'))
        load_net_clean = OrderedDict()  
        for k, v in load_net.items(): 
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v

        net.load_state_dict(load_net_clean, strict=True)

    net.eval()

    logging.info('start testing...')
    logging.info('%d testing samples' % (test_dataset.__len__()))

    PSNR_all = []
    PSNR_center = []
    PSNR_corner = []
    SSIM_all = []
    SSIM_center = []
    SSIM_corner = []

    with torch.no_grad():
        for batch_samples in tqdm(test_dataloader):

            image_name = batch_samples['name'][0]
            path = os.path.join(args.save_dir, image_name)

            batch_samples = prepare(batch_samples)

            LR_center = batch_samples['LR_center']
            Ref_SIFT = batch_samples['Ref_SIFT']
            LR = batch_samples['LR']

            #padding
            sh_im = LR.size()
            expanded_h = sh_im[-2] % args.chunk_size
            if expanded_h:
                expanded_h = args.chunk_size-expanded_h
            expanded_w = sh_im[-1] % args.chunk_size
            if expanded_w:
                expanded_w = args.chunk_size - expanded_w
            padexp = (0, expanded_w, 0, expanded_h)
            LR = F.pad(input=LR, pad=padexp, mode='reflect')

            LR = nn.ReplicationPad2d(8)(LR)

            #inference
            output = net(LR, LR_center, Ref_SIFT)

            #depadding
            if expanded_h:
                expanded_h = expanded_h*2
                output = output[:, :, :-expanded_h, :]
            if expanded_w:
                expanded_w = expanded_w*2
                output = output[:, :, :, :-expanded_w]


            output_save = output[0].flip(dims=(0,)).clamp(0., 1.)
            torchvision.utils.save_image(output_save, path)

            output_img = output[0].data.cpu().numpy().transpose(1, 2, 0)
            gt = batch_samples['HR'][0].data.cpu().numpy().transpose(1, 2, 0)

            corrdinate = corrdinates[image_name]
            PSNR_3 = calculate_psnr(output_img * 255, gt * 255, corrdinate)
            SSIM_3 = calculate_ssim(output_img * 255, gt * 255, corrdinate)


            PSNR_all.append(PSNR_3[0])
            PSNR_center.append(PSNR_3[1])
            PSNR_corner.append(PSNR_3[2])

            SSIM_all.append(SSIM_3[0])
            SSIM_center.append(SSIM_3[1])
            SSIM_corner.append(SSIM_3[2])

            logging.info('------------------------------------------------------------')
            logging.info(image_name)
            logging.info('all:     psnr: %.6f    ssim: %.6f' % (PSNR_3[0], SSIM_3[0]))
            logging.info('center:  psnr: %.6f    ssim: %.6f' % (PSNR_3[1], SSIM_3[1]))
            logging.info('corner:  psnr: %.6f    ssim: %.6f' % (PSNR_3[2], SSIM_3[2]))
            logging.info('------------------------------------------------------------')

            torch.cuda.empty_cache()

    PSNR_all_avg = np.mean(PSNR_all)
    PSNR_center_avg = np.mean(PSNR_center)
    PSNR_corner_avg = np.mean(PSNR_corner)
    SSIM_all_avg = np.mean(SSIM_all)
    SSIM_center_avg = np.mean(SSIM_center)
    SSIM_corner_avg = np.mean(SSIM_corner)

    logging.info('******Dataset Average******')
    logging.info('ALL:     psnr:%.06f   ssim:%.06f ' % (PSNR_all_avg, SSIM_all_avg))
    logging.info('CENTER:  psnr:%.06f   ssim:%.06f ' % (PSNR_center_avg, SSIM_center_avg))
    logging.info('CORNER:  psnr:%.06f   ssim:%.06f ' % (PSNR_corner_avg, SSIM_corner_avg))


if __name__ == '__main__':
    main()
