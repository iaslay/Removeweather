import argparse
import torch
import numpy as np
import random
import torch.nn.functional as F
from math import log10
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torchvision.utils as utils
import time
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Hyper-parameters for network')
    parser.add_argument('--learning_rate', help='Set the learning rate', default=2e-4, type=float)
    parser.add_argument('--train_path', default="../data/", type=str)
    parser.add_argument('--test_path', default="../data", type=str)
    parser.add_argument('--test_name', default="test_b", type=str)
    parser.add_argument('--crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
    parser.add_argument('--train_batch_size', help='Set the training batch size', default=16, type=int)
    parser.add_argument('--epoch_start', help='Starting epoch number of the training', default=0, type=int)
    parser.add_argument('--lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
    parser.add_argument('--val_batch_size', help='Set the validation/test batch size', default=1, type=int)
    parser.add_argument('--save_path', help='directory for saving the networks of the experiment', default="../save_data", type=str)
    parser.add_argument('--seed', help='set random seed', default=19, type=int)
    parser.add_argument('--device_idx', default=[9], nargs='+', type=int)
    parser.add_argument('--num_epochs', help='number of epochs', default=200, type=int)
    parser.add_argument('--lr_decay', default=0.5, type=float)
    parser.add_argument('--checkpoint', default=None, type=str)
    return init_args(parser.parse_args())

class Config(object):
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.crop_size = args.crop_size
        self.train_batch_size = args.train_batch_size
        self.epoch_start = args.epoch_start
        self.lambda_loss = args.lambda_loss
        self.val_batch_size = args.val_batch_size
        self.save_path = args.save_path
        self.num_epochs = args.num_epochs
        self.train_path = args.train_path
        self.test_path = args.test_path
        self.test_name = args.test_name
        self.device_idx = args.device_idx
        self.lr_decay = args.lr_decay
        self.checkpoint = args.checkpoint
        
def init_args(args):
    seed = args.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
    return Config(args)

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)


def calc_psnr(im1, im2):
    im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2], im2.shape[3], 3).detach().cpu().numpy()

    # im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    # im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_psnr(im1, im2, data_range=1)]
    # print(im1, im2)
    # ans = [10 * torch.log10(1 / F.mse_loss(im1, im2)).item()]
    # print(ans)

    return ans


def calc_ssim(im1, im2):
    im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2], im2.shape[3], 3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_ssim(im1_y, im2_y)]
    return ans


def to_psnr(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                          range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [compare_ssim(pred_image_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True) for ind
                 in range(len(pred_image_list))]

    return ssim_list


def validation(net, val_data_loader, device_idx, exp_name, save_tag=False):
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt, imgid = val_data
            input_im = input_im.cuda(device=device_idx[0])
            gt = gt.cuda(device=device_idx[0])
            pred_image = net(input_im)
        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))
        # print(pred_image)
        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            # print()
            save_image(pred_image, imgid, exp_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def validation_val(net, val_data_loader, device_idx, exp_name, save_tag=False):
    psnr_list = []
    ssim_list = []
    test_bar = tqdm(val_data_loader, initial=1, dynamic_ncols=True)
    for val_data in test_bar:

        with torch.no_grad():
            input_im, gt, imgid = val_data
            input_im = input_im.cuda(device=device_idx[0])
            gt = gt.cuda(device=device_idx[0])
            pred_image = net(input_im)
            

        # --- Calculate the average PSNR --- #
        psnr = calc_psnr(pred_image, gt)
        psnr_list.extend(psnr)

         #--- Calculate the average SSIM --- #
        ssim = calc_ssim(pred_image, gt)
        ssim_list.extend(ssim)
        
        
        with open('{}/testing_log.txt'.format(exp_name), 'a') as f:
          print(f'name{imgid}, PSNR={psnr}, SSIM={ssim}', file=f)

        # --- Save image --- #
        if save_tag:
            # print()
            save_image(pred_image, imgid, exp_name)
        test_bar.set_description(
            'Test Iter:PSNR:{:.2f} SSIM{:.3f}'.format(np.mean(psnr_list), np.mean(ssim_list)))
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def validation_va1(net, val_data_loader, device_idx, exp_name, save_tag=False):
    test_bar = tqdm(val_data_loader, initial=1, dynamic_ncols=True)
    for val_data in test_bar:

        with torch.no_grad():
            input_im,imgid = val_data
            input_im = input_im.cuda(device=device_idx[0])
            pred_image = net(input_im)
        # --- Save image --- #
        if save_tag:
            # print()
            save_image(pred_image, imgid, exp_name)
    return


def save_image(pred_image, image_name, save_path):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)

    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]
       # print(image_name_1)
        utils.save_image(pred_image_images[ind], '{}/picture/{}'.format(save_path, image_name_1))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, save_path):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))
    # --- Write the training log --- #
    with open('{}/training_log.txt'.format(save_path), 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)


def adjust_learning_rate(optimizer, epoch, lr_decay=0.3):
    # --- Decay learning rate --- #
    step = 100

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
