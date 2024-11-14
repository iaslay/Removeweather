import torch.optim
import torchvision
from torchvision.models import vgg16
from utils import parse_args, LossNetwork, to_psnr, print_log, validation, adjust_learning_rate
import data_set
from torch.utils.data import DataLoader
import model

import time
import itertools

import torch.nn.functional as F
from tqdm import tqdm

if __name__ == '__main__':
    args = parse_args()
    device_idx = args.device_idx
    print(device_idx[0])
    test_dataset = data_set.TestData(args.crop_size, args.test_path, args.test_name)
    train_dataset = data_set.TrainData(args.crop_size, args.train_path)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, num_workers=8)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers= 8)

    net = model.Removeweather()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    net = torch.nn.DataParallel(net, device_ids=device_idx)
    net = net.cuda(device=device_idx[0])

    if args.checkpoint:
      net.load_state_dict(torch.load('{}/latest.pth'.format(args.checkpoint)))
    # --- Define the perceptual loss network --- #
    vgg_model = vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[:16]
    vgg_model = vgg_model.cuda(device=device_idx[0])
    for param in vgg_model.parameters():
        param.requires_grad = False

    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    net.eval()
    old_val_psnr1, old_val_ssim1 = validation(net, test_loader, device_idx, args.save_path)
    print('Rain Drop old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr1, old_val_ssim1))

    net.train()
    last = None
    for epoch in range(args.epoch_start, args.num_epochs):
        psnr_list = []
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, lr_decay=args.lr_decay)
        total_loss, count = 0., 0
        train_bar = tqdm(train_loader, initial=1, dynamic_ncols=True)
        for train_data in train_bar:
            input_image, gt, imgid = train_data
            input_image = input_image.cuda(device=device_idx[0])
            gt = gt.cuda(device=device_idx[0])
            count += 1
            # --- Zero the parameter gradients --- #
            optimizer.zero_grad()

            net.train()
            pred_image = net(input_image)

            smooth_loss = F.smooth_l1_loss(pred_image, gt)
            # 频域损失
            smooth_loss_ = F.l1_loss(torch.fft.rfft2(pred_image),  torch.fft.rfft2(gt))
            perceptual_loss = loss_network(pred_image, gt)
            loss = smooth_loss + args.lambda_loss * perceptual_loss+smooth_loss_*args.lambda_loss*0.1
            # print(smooth_loss, perceptual_loss)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # --- To calculate average PSNR --- #
            psnr_list.extend(to_psnr(pred_image, gt))
            train_bar.set_description("loss:{:.2f}".format(total_loss/count))
            

            # --- Calculate the average training PSNR in one epoch --- #
        train_psnr = sum(psnr_list) / len(psnr_list)

        # --- Save the network parameters --- #
        torch.save(net.state_dict(), '{}/latest.pth'.format(args.save_path))
        # --- Use the evaluation model in testing --- #
        net.eval()
        val_psnr1, val_ssim1 = validation(net, test_loader, device_idx, args.save_path)
        one_epoch_time = time.time() - start_time
        print("Rain Drop")
        print_log(epoch + 1,args.num_epochs, one_epoch_time, train_psnr, val_psnr1, val_ssim1, args.save_path)
        if val_psnr1 >= old_val_psnr1:
            torch.save(net.state_dict(), '{}/best.pth'.format(args.save_path))
            old_val_psnr1 = val_psnr1