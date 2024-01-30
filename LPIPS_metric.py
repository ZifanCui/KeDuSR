import torch
import lpips
# from IPython import embed
import os
import glob
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='referenceSR Training')
    parser.add_argument('--SR_dir', default='./inference_result/DuSR-Real/DuSR-Real_220000', type=str) #result
    parser.add_argument('--HR_dir', default='/dataset/DuSR-Real/test/HR/', type=str)                   #HR(GT)
    args = parser.parse_args()

    use_gpu = True         # Whether to use GPU
    spatial = True         # Return a spatial map of perceptual distance.

    # Linearly calibrated models (LPIPS)
    loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
    # loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

    if(use_gpu):
        loss_fn.cuda()
        
    SRs = glob.glob(os.path.join(args.SR_dir, '*'))
    SRs.sort()

    HRs = glob.glob(os.path.join(args.HR_dir, '*'))
    HRs.sort()

    dist_ = []

    for SR, HR in tqdm(zip(SRs, HRs)):
        dummy_im0 = lpips.im2tensor(lpips.load_image(SR))                            
        dummy_im1 = lpips.im2tensor(lpips.load_image(HR))
        if(use_gpu):
            dummy_im0 = dummy_im0.cuda()
            dummy_im1 = dummy_im1.cuda()
        dist = loss_fn.forward(dummy_im0, dummy_im1)
        dist_.append(dist.mean().item())
    print('Avarage LPIPS: %.3f' % (sum(dist_)/len(SRs)))



