import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import cv2
import glob
from tqdm import tqdm


class TrainSet(Dataset):
    def __init__(self, args):

        self.args = args

        LR_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'train/LR', '*')))
        HR_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'train/HR', '*')))
        Ref_full_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'train/Ref_full', '*')))
        self.scale = args.sr_scale

        self.LR_imgs = []
        self.HR_imgs = []
        self.Ref_full_imgs = []

        for i in tqdm(range(len(LR_list))):
            self.LR_imgs.append(cv2.imread(LR_list[i], -1))
            self.HR_imgs.append(cv2.imread(HR_list[i], -1))
            self.Ref_full_imgs.append(cv2.imread(Ref_full_list[i], -1))


    def __len__(self):
        return len(self.LR_imgs) * 50
    

    def crop_patch(self, LR, HR, Ref_full, p):
        ih, iw = LR.shape[:2]
        pw = random.randint(0, iw - p)
        ph = random.randint(0, ih - p)
        hpw, hph = self.scale * pw, self.scale * ph
        hr_patch_size = self.scale * p

        #During training, extracting patches around LR as LR-center, termed as LR_nearby
        nearby_h  = random.randint(32, 96) * random.choice((-1, 1)) 
        nearby_w  = random.randint(32, 96) * random.choice((-1, 1))
        ph_nearby = ph + nearby_h
        pw_nearby = pw + nearby_w
        ph_nearby = max(0 + 10, min(ph_nearby, ih - p -10))
        pw_nearby = max(0 + 10, min(pw_nearby, iw - p -10))


        hpw_nearby, hph_nearby = self.scale * pw_nearby, self.scale * ph_nearby

        #return lr  hr  lr_nearby  ref
        return LR[ph:ph+p, pw:pw+p], \
			   HR[hph:hph+hr_patch_size, hpw:hpw+hr_patch_size], \
               LR[ph_nearby:ph_nearby+p, pw_nearby:pw_nearby+p], \
               Ref_full[hph_nearby :hph_nearby+hr_patch_size , hpw_nearby :hpw_nearby+hr_patch_size]

                
    def augment(self, *args, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        k1 = np.random.randint(0, 3)
        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]        
            
            img = np.rot90(img, k1)
            
            return img

        return [_augment(a) for a in args]


    def __getitem__(self, idx):

        idx = idx % len(self.LR_imgs)

        LR = self.LR_imgs[idx]
        HR = self.HR_imgs[idx]
        Ref_full = self.Ref_full_imgs[idx]


        lr, hr, lr_nearby, ref = self.crop_patch(LR, HR, Ref_full, p=self.args.patch_size)
        lr, hr, lr_nearby, ref = self.augment(lr, hr, lr_nearby, ref)

        sample = {
                    'lr': lr,
                    'lr_nearby': lr_nearby,
                    'ref': ref,
                    'hr': hr,
                }

        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        return sample

    
class TestSet_cache(Dataset):
    def __init__(self, args):
        LR_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'test/LR', '*')))
        HR_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'test/HR', '*')))
        LR_center_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'test/LR_center', '*')))
        Ref_SIFT_list = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset_name, 'test/Ref_SIFT', '*')))
        

        self.scale = args.sr_scale


        self.LR_imgs = []
        self.HR_imgs = []
        self.LR_center_imgs = []
        self.Ref_SIFT_imgs = []
        self.names = []


        for i in tqdm(range(len(LR_list))):

            self.names.append(os.path.basename(LR_list[i]))
            self.LR_imgs.append(cv2.imread(LR_list[i], -1))
            self.HR_imgs.append(cv2.imread(HR_list[i], -1))
            self.LR_center_imgs.append(cv2.imread(LR_center_list[i], -1))
            self.Ref_SIFT_imgs.append(cv2.imread(Ref_SIFT_list[i], -1))


    def __len__(self):
        return len(self.LR_imgs)


    def __getitem__(self, idx):

        sample = {'LR': self.LR_imgs[idx],
                 'HR': self.HR_imgs[idx],               
                  'LR_center': self.LR_center_imgs[idx],
                  'Ref_SIFT': self.Ref_SIFT_imgs[idx],
                  }

        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        sample['name'] = self.names[idx]

        return sample



