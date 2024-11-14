import time
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from data_set import TestData
from model import Removeweather
from utils import validation, validation_val
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('--val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('--save_path', help='directory for saving the networks of the experiment', type=str,default='../result/test1')
parser.add_argument('--crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('--seed', help='set random seed', default=19, type=int)
parser.add_argument('--device_idx', default=[0], nargs='+', type=int)
parser.add_argument('--checkpoint', default='../save_data/e_200s_200', type=str)
args = parser.parse_args()

val_batch_size = args.val_batch_size
save_path = args.save_path
torch.cuda.set_device(args.device_idx[0])
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

val_data_dir = '../data'
val_filename = 'test1'
device_idx = args.device_idx
val_data_loader = DataLoader(TestData(args.crop_size, val_data_dir, val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)

net = Removeweather()
net = net.cuda(device_idx[0])
net = torch.nn.DataParallel(net, device_ids=device_idx)
net.load_state_dict(torch.load('{}/best.pth'.format(args.checkpoint)))
net.eval()
print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim = validation_val(net, val_data_loader, device_idx, save_path, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))
