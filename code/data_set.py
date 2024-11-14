import glob
import os
from random import randrange
import torch.utils.data as data
from PIL import Image
from random import random
from torchvision.transforms import Compose, ToTensor, Normalize
import re
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TrainData(data.Dataset):
    def __init__(self, crop_size, datapath):
        super(TrainData, self).__init__()
        self.gt_names = sorted(glob.glob('{}/allweather/gt/*'.format(datapath)))
        self.input_names = sorted(glob.glob('{}/allweather/input/*'.format(datapath)))
        self.crop_size = crop_size
        self.train_data_dir = datapath

    def __getitem__(self, item):
        num_to_train = 568
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[item]
        gt_name = self.gt_names[item]
        img_id = re.split('/', input_name)[-1][6:-4]
        input_img = Image.open(input_name)
        try:
            gt_img = Image.open(gt_name)
        except:
            gt_img = Image.open(gt_name).convert('RGB')
        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)
        width, height = input_img.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        # # --- Check the channel is 3 or not --- #
        # if list(input_im.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
        #     raise Exception('Bad image channel: {}'.format(gt_name))

        return input_im, gt, img_id

    def __len__(self):
        return len(self.input_names)


class TestData(data.Dataset):
    def __init__(self, crop_size, datapath, filename):
        super(TestData, self).__init__()
        self.gt_names = sorted(glob.glob('{}/{}/{}/gt/*'.format(datapath, filename,filename)))
        self.input_names = sorted(glob.glob('{}/{}/{}/input/*'.format(datapath,filename, filename)))
        self.crop_size = crop_size
        self.train_data_dir = datapath

    def __getitem__(self, item):
        input_name = self.input_names[item]
        gt_name = self.gt_names[item]
        input_img = Image.open(input_name)
        gt_img = Image.open(gt_name)

        # Resizing image in the multiple of 16"
        wd_new, ht_new = input_img.size
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        input_img = input_img.resize((wd_new, ht_new), Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __len__(self):
        return len(self.input_names)

class TestData1(data.Dataset):
    def __init__(self, crop_size, datapath):
        super(TestData1, self).__init__()
        self.input_names = sorted(glob.glob('{}/*'.format(datapath)))
        self.crop_size = crop_size
        self.train_data_dir = datapath

    def __getitem__(self, item):
        input_name = self.input_names[item]
        input_img = Image.open(input_name)

        # Resizing image in the multiple of 16"
        wd_new, ht_new = input_img.size
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        input_img = input_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_im = transform_input(input_img)

        return input_im, input_im,input_name

    def __len__(self):
        return len(self.input_names)