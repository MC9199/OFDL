#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import shutil
import random
import argparse
import numpy as np
from math import ceil
from pathlib import Path
from scipy.io import loadmat
from skimage import img_as_float32
from networks.generators import GeneratorState, GeneratorRain

# torch package
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

torch.set_default_dtype(torch.float32)

class trainer:
    def __init__(self, args):
        # setting random seed
        print('*'*80)
        print('Setting random seed: {:d}...'.format(args['seed']))
        self.seed = args['seed']
        self.set_seed()

        # setting gpu
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args['gpu_id']))

        # prepare data
        self.rain_path =args['rain_path']
        self.prepare_data()                           # self.rain_data: n x 1 x h x w, float32

        # some necessary parameters
        self.resume_path = args['resume']
        self.log_dir = args['log_dir']
        self.model_dir = args['model_dir']

    def set_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_data(self):
        # rain_data = loadmat(self.rain_path)['rain_part'][:, :, :150]    # h x w x n, uint8
        # self.frames = rain_data.shape[-1]
        # self.rain_data = img_as_float32(rain_data.transpose([2,0,1])[:, np.newaxis,])

        rain_data = loadmat(self.rain_path)['rain_layer']    # n x c x h x w, uint8
        _, _, h, w = rain_data.shape
        self.frames = rain_data.shape[0]
        self.rain_data = img_as_float32(rain_data[:, :, :int(h/2), :int(w/2)])

    def check_resume(self):
        if self.resume_path is not None:
            print('Loading checkpoint from {:s}'.format(self.resume_path))
            checkpoint = torch.load(self.resume_path)
            self.current_epoch = checkpoint['epoch']
            self.log_im_step = checkpoint['step_img']
            self.GStateNet.load_state_dict(checkpoint['GState'])
            self.GRainNet.load_state_dict(checkpoint['GRain'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for ii in range(self.current_epoch):
                self.decay_lr(ii)
        else:
            self.current_epoch = 0

            # path to save model
            if Path(self.model_dir).is_dir():
                shutil.rmtree(str(Path(self.model_dir)))
            Path(self.model_dir).mkdir()

    def open_tensorboard(self):
        if self.resume_path is None:
            if Path(self.log_dir).is_dir():
                shutil.rmtree(str(Path(self.log_dir)))
            Path(self.log_dir).mkdir()

        self.writer = SummaryWriter(str(Path(self.log_dir)))

        if self.resume_path is None:
            self.log_im_step = 0

    def G_forward_truncate(self, truncate_Z, initial_state, motion_type=None):
        '''
        Forward propagation of Generator for truncated data.
        :param truncate_Z: num_frame x latent_size tensor
        :param initial_state:  1 x state_size tensor
        :param motion_type:  1 x state_size tensor or None
        '''
        state_all = []
        state_next = initial_state
        for ii in range(truncate_Z.shape[0]):
            input_Z = truncate_Z[ii, :].view([1,-1])
            state_next = self.GStateNet(input_Z, state_next, motion_type) # 1 x state_size
            state_all.append(state_next)

        rain_gen = self.GRainNet(torch.cat(state_all, dim=0))             # n x c x p x p

        return rain_gen, state_next

    def get_loss(self, rain_gt, rain_gen):
        '''
        :param rain_gt: num_frame x c x p x p tensor, rain layer groundtruth
        :param rain_gen: num_frame x c x p x p tensor, generated rain
        '''
        sigma = (rain_gt - rain_gen.detach()).flatten().std().item()
        loss = 0.5 / (sigma**2) * (rain_gt - rain_gen).square().sum()
        loss /= rain_gt.shape[0]

        return loss

    def freeze_Generator(self):
        for param in self.GStateNet.parameters():
            param.requires_grad = False
        for param in self.GRainNet.parameters():
            param.requires_grad = False

    def unfreeze_Generator(self):
        for param in self.GStateNet.parameters():
            param.requires_grad = True
        for param in self.GRainNet.parameters():
            param.requires_grad = True

    def save_model(self, ii):
        '''
        Index of current_epoch
        '''
        model_prefix = 'model_'
        save_path_model = str(Path(self.model_dir) / (model_prefix+str(ii+1)))
        torch.save({'epoch': ii+1,
                    'step_img': self.log_im_step,
                    'GStateNet': self.GStateNet.state_dict(),
                    'GRainNet': self.GRainNet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, save_path_model)
        model_state_prefix = 'model_state_'
        save_path_model_state = str(Path(self.model_dir) / (model_state_prefix+str(ii+1)+'.pt'))
        torch.save({'GState': self.GStateNet.state_dict(),
                    'GRain': self.GRainNet.state_dict(),
                    'Z': self.Z,
                    'S': self.S}, save_path_model_state)

    def train(self, args):
        # build the network
        self.GStateNet = GeneratorState(latent_size=args['latent_size'],
                                        state_size=args['state_size'],
                                        motion_size=args['motion_size'],
                                        num_feature=args['feature_state']).cuda()
        self.GRainNet = GeneratorRain(im_size=self.rain_data.shape[2:],
                                      out_channels=args['out_channels'],
                                      state_size=args['state_size'],
                                      num_feature=args['feature_rain_G']).cuda()

        # optimizer
        self.optimizer = optim.Adam([{'params': self.GStateNet.parameters(),
                                      'lr': args['lr_GState'],
                                      'weight_decay': 0},
                                     {'params': self.GRainNet.parameters(),
                                      'lr': args['lr_GRain'],
                                      'weight_decay': 0}], betas=(0.5, 0.999))

        # initialize the latent and state variables
        self.Z = torch.randn([self.frames, args['latent_size']]).cuda()
        self.S = torch.randn([1, args['state_size']]).cuda()

        # resume from some specific epoch
        self.check_resume()

        # open tensorboard
        self.open_tensorboard()

        print('*'*80)
        print('Begin training...')
        for ii in range(self.current_epoch, args['epochs']):
            lossM_epoch = 0
            for tt in range(ceil(self.frames / args['truncate'])):
                start = tt * args['truncate']
                end = min((tt+1)*args['truncate'], self.frames)
                t_slice = slice(start, end)
                rain_gt = torch.from_numpy(self.rain_data[t_slice,]).cuda()
                input_Z = self.Z[t_slice,]
                if tt == 0:
                    input_S = self.S
                else:
                    input_S = torch.zeros_like(state_next, requires_grad=False).copy_(state_next.data)
                for _ in range(args['max_iter_EM']):
                    # M-step
                    self.optimizer.zero_grad()
                    rain_gen, _ = self.G_forward_truncate(input_Z, input_S)
                    lossM = self.get_loss(rain_gt, rain_gen)
                    lossM.backward()
                    self.optimizer.step()
                    lossM_epoch += lossM.item()

                    # E-step
                    self.freeze_Generator()
                    for ss in range(args['langevin_steps']):
                        input_Z.requires_grad = True
                        if tt == 0:
                            input_S.requires_grad = True
                        rain_gen_E, state_next = self.G_forward_truncate(input_Z, input_S)
                        lossE = self.get_loss(rain_gt, rain_gen_E)
                        lossE.backward()
                        if tt == 0:
                            input_S = input_S - 0.5 * (args['delta']**2) * (input_S.grad + input_S)
                            if ss < (args['langevin_steps']/3):
                                input_S = input_S + args['delta'] * torch.randn_like(input_S)
                            input_S.detach_()
                        input_Z = input_Z - 0.5 * (args['delta']**2) * (input_Z.grad + input_Z/input_Z.shape[0])
                        if ss < (args['langevin_steps']/3):
                            input_Z = input_Z + args['delta'] * torch.randn_like(input_Z)
                        input_Z.detach_()
                    self.unfreeze_Generator()
                    self.Z[t_slice,] = input_Z.data
                    if tt == 0:
                        self.S = input_S.data

                # add tensorboard
                x1 = vutils.make_grid(rain_gen, normalize=True, scale_each=True)
                self.writer.add_image('Generated Rain', x1, self.log_im_step)
                x2 = vutils.make_grid(rain_gt, normalize=True, scale_each=True)
                self.writer.add_image('Groundtruth Rain', x2, self.log_im_step)
                self.log_im_step += 1

            lossM_epoch /= ((tt+1)*args['max_iter_EM'])

            # print log
            lr_GState = self.optimizer.param_groups[0]['lr']
            lr_GRain = self.optimizer.param_groups[1]['lr']
            print('Epoch:{:02d}/{:02d}, LossM:{:.4e}, lrGSR:{:.2e}/{:.2e}'.format(ii+1, args['epochs'],
                                                                    lossM_epoch, lr_GState, lr_GRain))

            # save model
            self.save_model(ii)

            # close the tensorboard
            self.writer.close()

