import os
import json
from tqdm import tqdm
import logging
import math
import numpy as np
import importlib
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
import importlib
from dataloader.dataset import TrainSet
from crop_psnr import calculate_ssim, calculate_psnr
from models.losses import PerceptualLoss, AdversarialLoss, CharbonnierLoss
from models.archs.KeDuSR_arch import KeDuSR


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.device = torch.device('cuda')

        # init train dataloader
        self.train_dataset = TrainSet(self.args)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

        # init test dataloader
        testset = getattr(importlib.import_module('dataloader.dataset'), args.testloader, None)
        self.test_dataset = testset(args)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, num_workers=1, shuffle=False)

        #LR-center corrdinate
        with open('./corrdinate/corrdinate_{}.json'.format(args.dataset_name),'r') as f:
            self.corrdinates = json.load(f)


        self.net = KeDuSR(args).cuda()
        logging.info('----- generator parameters: %f -----' % (sum(param.numel() for param in self.net.parameters()) / (10**6)))

        if args.resume:
            self.load_networks('net', args.resume)

        
        if args.loss_Charbonnier:
            self.criterion_Charbonnier = CharbonnierLoss().to(self.device)
            self.lambda_Charbonnier = args.lambda_Charbonnier
            logging.info('  using Charbonnier loss...')

        if args.loss_adv:
            self.criterion_adv = AdversarialLoss(gan_k=1)
            self.lambda_adv = args.lambda_adv
            logging.info('  using adv loss...')

        if args.loss_perceptual:
            self.criterion_perceptual = PerceptualLoss(layer_weights={'conv5_4': 1.}).to(self.device)
            self.lambda_perceptual = args.lambda_perceptual
            logging.info('  using perceptual loss...')

        self.optimizer_G = torch.optim.Adam(self.net.parameters(), lr=args.lr)


        #CosineAnnealing 
        t = self.args.warm_up_iter   # t=10000#10k
        T = self.args.total_iter     # T=250000#250k
        decline_rate = 1e-2 #最小值
        n_t=0.5       #默认0.5
        lambda1 = lambda epoch: (0.9*epoch / t+0.1) if epoch < t else  decline_rate  if (n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))<decline_rate) or (epoch>T) else n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda1)


        if args.resume_optim:
            self.load_networks('optimizer_G', args.resume_optim)
        if args.resume_scheduler:
            self.load_networks('scheduler', args.resume_scheduler)


    def prepare(self, batch_samples):
        for key in batch_samples.keys():
            if 'name' not in key:
                batch_samples[key] = batch_samples[key].to(self.device)
        return batch_samples


    def train(self):
        logging.info('training on  ...' + self.args.dataset_name)
        logging.info('%d training samples' % (self.train_dataset.__len__()))
        logging.info('the init lr: %f'%(self.args.lr))
        # steps = 0
        self.net.train()

        if self.args.use_tb_logger:
            tb_logger = SummaryWriter(log_dir= self.args.save_folder + '/tb_logger/')

        self.best_psnr = 0
        self.best_ssim = 0
        # self.augmentation = False  # disenable data augmentation to warm up the encoder
             
        if self.args.resume != '':
            current_iter = int(self.args.resume.split('_')[-1][:-4])
        else:
            current_iter = 0

        loss_Charbonnier_p = 0
        loss_perceptual_p = 0
        loss_adv_p = 0
        loss_d_p = 0
        for idx_i in range(1000000):

            #training over
            if current_iter > self.args.total_iter:
                break
            
            for batch_samples in self.train_dataloader:
                current_iter += 1
                log_info = 'epoch:%03d step:%04d lr:%.06f  ' % (idx_i, current_iter, self.optimizer_G.param_groups[0]['lr'])

                batch_samples = self.prepare(batch_samples)

                #KeDuSR
                output = self.net(batch_samples['lr'], batch_samples['lr_nearby'], batch_samples['ref'])


                loss = 0
                self.optimizer_G.zero_grad()
                

                if self.args.loss_Charbonnier:
                    Charbonnier_loss = self.criterion_Charbonnier(output, batch_samples['hr'])
                    Charbonnier_loss = Charbonnier_loss * self.lambda_Charbonnier
                    loss_Charbonnier_p += Charbonnier_loss.item()
                    loss += Charbonnier_loss

                if self.args.loss_perceptual and current_iter > self.args.warm_up_iter:
                    perceptual_loss, _ = self.criterion_perceptual(output, batch_samples['hr'])
                    perceptual_loss = perceptual_loss * self.lambda_perceptual
                    loss_perceptual_p += perceptual_loss.item()
                    loss += perceptual_loss

                if self.args.loss_adv and current_iter > self.args.warm_up_iter:
                    adv_loss, d_loss = self.criterion_adv(output, batch_samples['hr'])
                    adv_loss = adv_loss * self.lambda_adv
                    loss_adv_p += adv_loss.item()
                    loss_d_p += d_loss.item()
                    loss += adv_loss


                loss.backward()
                self.optimizer_G.step()
                self.scheduler.step()

                #logging
                if current_iter % self.args.log_freq == 0:
                    loss_Charbonnier_p = loss_Charbonnier_p / self.args.log_freq
                    log_info += 'Ch_loss:%.06f ' % (loss_Charbonnier_p)

                    if self.args.loss_perceptual:
                        loss_perceptual_p = loss_perceptual_p / self.args.log_freq
                        loss_adv_p = loss_adv_p / self.args.log_freq
                        loss_d_p = loss_d_p / self.args.log_freq

                        log_info += 'perceptual_loss:%.06f ' % (loss_perceptual_p)
                        log_info += 'adv_loss:%.06f ' % (loss_adv_p)
                        log_info += 'd_loss:%.06f ' % (loss_d_p)

                        tb_logger.add_scalars(
                            main_tag='loss_split',
                            tag_scalar_dict={
                                'Ch_loss': loss_Charbonnier_p,
                                'perceptual_loss': loss_perceptual_p,
                                'adv_loss': loss_adv_p,
                                'd_loss': loss_d_p
                            },
                            global_step=current_iter
                        )
                        
                    logging.info(log_info)
                    tb_logger.add_scalar('Ch_loss', loss_Charbonnier_p, current_iter)
                        
                    loss_Charbonnier_p = 0
                    loss_perceptual_p = 0
                    loss_adv_p = 0
                    loss_d_p = 0


 
                #evaluate
                if current_iter % self.args.test_freq == 0:

                    logging.info('Saving state, epoch: %d iter:%d' % (idx_i, current_iter))
                    self.save_networks('net', current_iter)
                    self.save_networks('optimizer_G', current_iter)
                    self.save_networks('scheduler', current_iter)

                    PSNR_all_avg, PSNR_center_avg, PSNR_corner_avg, SSIM_all_avg, SSIM_center_avg, SSIM_corner_avg = self.evaluate(current_iter)
                    tb_logger.add_scalar('PSNR', PSNR_all_avg, current_iter)
                    tb_logger.add_scalar('SSIM', SSIM_all_avg, current_iter)

                    tb_logger.add_scalars(main_tag='PSNR_3',
                                          tag_scalar_dict={'PSNR_all': PSNR_all_avg,
                                                        'PSNR_center': PSNR_center_avg,
                                                        'PSNR_corner': PSNR_corner_avg,
                                                        },
                                          global_step=current_iter)
                    
                    tb_logger.add_scalars(main_tag='SSIM_3',
                                          tag_scalar_dict={'SSIM_all': SSIM_all_avg,
                                                        'SSIM_center': SSIM_center_avg,
                                                        'SSIM_corner': SSIM_corner_avg,
                                                        },
                                          global_step=current_iter)
                    
                    logging.info('*******************Dataset Average ' + str(current_iter) + '*******************')                    
                    logging.info('ALL:     psnr:%.06f   ssim:%.06f ' % (PSNR_all_avg, SSIM_all_avg))
                    logging.info('CENTER:  psnr:%.06f   ssim:%.06f ' % (PSNR_center_avg, SSIM_center_avg))
                    logging.info('CORNER:  psnr:%.06f   ssim:%.06f ' % (PSNR_corner_avg, SSIM_corner_avg))

                    self.net.train()


                    #save best
                    if SSIM_all_avg > self.best_ssim:
                        self.best_ssim = SSIM_all_avg

                        logging.info('best_ssim:%.06f ' % (self.best_ssim))


                    if PSNR_all_avg > self.best_psnr:
                        self.best_psnr = PSNR_all_avg

                        logging.info('best_psnr:%.06f ' % (self.best_psnr))
                        logging.info('Saving state, epoch: %d iter:%d' % (idx_i, current_iter))
                        self.save_networks('net', 'best')
                        self.save_networks('optimizer_G', 'best')
                        self.save_networks('scheduler', 'best')
                    logging.info('**************************************')


        ## end of training
        tb_logger.close()
        logging.info('The training stage on is over!!!')


    def evaluate(self, current_iter):
        self.net.eval()
        logging.info('start testing...')
        logging.info('%d testing samples' % (self.test_dataset.__len__()))

        PSNR_all = []
        PSNR_center = []
        PSNR_corner = []
        SSIM_all = []
        SSIM_center = []
        SSIM_corner = []
        with torch.no_grad():
            for batch_samples in tqdm(self.test_dataloader):

                image_name = batch_samples['name'][0]
                save_path = os.path.join(self.args.save_folder, '{}_{}'.format('KeDuSR',current_iter))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                path = os.path.join(save_path, image_name)

                output = self.patch_inference(batch_samples)
                output_save = output[0].flip(dims=(0,)).clamp(0., 1.)
                torchvision.utils.save_image(output_save, path)

                output_img = output[0].data.cpu().numpy().transpose(1, 2, 0)
                gt = batch_samples['HR'][0].data.cpu().numpy().transpose(1, 2, 0)


                corrdinate = self.corrdinates[image_name]
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

        return PSNR_all_avg, PSNR_center_avg, PSNR_corner_avg, SSIM_all_avg, SSIM_center_avg, SSIM_corner_avg


    def load_networks(self, net_name, resume, strict=True):
        load_path = resume
        network = getattr(self, net_name)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel): #"Processing ".module"
            network = network.module     
        load_net = torch.load(load_path, map_location=torch.device('cuda'))
        load_net_clean = OrderedDict()  
        for k, v in load_net.items():   
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        if ('optimizer' in net_name) or ('scheduler' in net_name):
            network.load_state_dict(load_net_clean)
        else:
            network.load_state_dict(load_net_clean, strict=True)


    def save_networks(self, net_name, current_iter):
        network = getattr(self, net_name)
        save_filename = '{}_{}.pth'.format(net_name, current_iter)
        save_path = os.path.join(self.args.snapshot_save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        if not 'optimizer' and not 'scheduler' in net_name:
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)


    def patch_inference(self, batch_samples):

        batch_samples = self.prepare(batch_samples)

        LR_center = batch_samples['LR_center']
        Ref_SIFT = batch_samples['Ref_SIFT']
        LR = batch_samples['LR']


        #padding
        sh_im = LR.size()
        expanded_h = sh_im[-2] % self.args.chunk_size

        if expanded_h:
            expanded_h = self.args.chunk_size-expanded_h
        expanded_w = sh_im[-1] % self.args.chunk_size
        if expanded_w:
            expanded_w = self.args.chunk_size - expanded_w

        padexp = (0, expanded_w, 0, expanded_h)
        LR = F.pad(input=LR, pad=padexp, mode='reflect')

        LR = nn.ReplicationPad2d(8)(LR)  #torch.Size([1, 3, 448, 896]) torch.Size([1, 3, 464, 912])


        # inference
        output = self.net(LR, LR_center, Ref_SIFT)


        #depadding
        if expanded_h:
            expanded_h = expanded_h*2
            output = output[:, :, :-expanded_h, :]
        if expanded_w:
            expanded_w = expanded_w*2
            output = output[:, :, :, :-expanded_w]

        return output



    