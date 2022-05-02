""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch

# pylint: disable=C0103,C0301,R0903,W0622

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataset', default='caltech101', help='folder | caltech101 | coco ')
        self.parser.add_argument('--dataroot', default='', help='path to dataset')        
        self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--icrop', type=int, default=224, help='input image crop size.')
        self.parser.add_argument('--isize', type=int, default=256, help='input image size.')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='resnet18', help='chooses which model to use. resnet18 | glp | inception_v3')
        self.parser.add_argument('--verbose', action='store_true', help='Print the training and model details.')
        self.parser.add_argument('--add_gussianblur', action='store_true', help='add_gussianblur to the transforms or not.')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--seed', default=42, type=int, help='manual seed')

        ##
        # Train
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train for')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum term of sgd')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--step_size', type=int, default='10', help='number of stepsize to lr decay')
        self.parser.add_argument('--gamma', type=int, default=50, help='multiply by a gamma every stepsize iterations')
        self.parser.add_argument('--num_classes', type=int, help='number of classes')

        #two_stream model
        self.parser.add_argument('--glp_path', type=str, default='', help='path to trained glp model for loading in two_stream network')
        self.parser.add_argument('--two_stream_path', type=str, default='', help='path to trained two_stream model for loading test on attacks only')
        self.parser.add_argument('--pretrained_path', type=str, default='', help='path to trained pretrained models for loading test on attacks only')
        self.parser.add_argument('--repeat_on_attacks', type=int, default=5, help='number of repeatition of attacks')
        self.parser.add_argument('--save_csv', type=str, default='results.csv', help='path to save results')
        
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.verbose:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'checkpoints')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
    
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
