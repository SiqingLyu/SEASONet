# -*- coding: UTF-8 -*-
'''
This is the training program of the SEASONet.
Change the network settings in ../configs/SEASONet.yml and run this file to train the network
The network structure is defined in ../models/SEASONet.py
'''
import os
import yaml
import shutil
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from functools import reduce
import sys
from torch.utils import data
from models import get_model
from utils import get_logger
from torch.autograd import Variable
import math
from tensorboardX import SummaryWriter #change tensorboardX
from dataloaders.dataloaders import *
from tools import *
from pytorch_tools import *
import gc
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append('/SEASONet')
torch.set_num_threads(3)


def add(num1, num2):
    return num1 + num2


def main(cfg, writer, logger, ckptsavedir):
    '''
    :param cfg: yaml file that contains the network settings
    :param writer: SummaryWriter for later training visualization
    :param logger: logger file to record the training log information
    :param ckptsavedir: the path to save the checkpoint
    '''
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 33))
    torch.cuda.manual_seed(cfg.get("seed", 33))
    np.random.seed(cfg.get("seed", 33))
    random.seed(cfg.get("seed", 33))
    # Setup device
    device = torch.device(cfg["training"]["device"])
    # Setup Dataloader
    data_path = cfg["data"]["path"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    # epochs = 286
    shuffle = cfg['training']['shuffle']
    buffer_assist = cfg['data']['buffer_assist']

    # Load dataset
    if buffer_assist:
        trainimg, trainlab, valimg, vallab, _, _, trainpre, valpre, _ = make_dataset(data_path, split=[0.8, 0.2, 0.0],
                                                                                 test_only=False, buffer_assist=cfg['data']['buffer_assist'])
    else:
        trainimg, trainlab, valimg, vallab, _, _ = make_dataset(data_path, split=[0.8, 0.2, 0.0], test_only=False)
        trainpre = None
        valpre = None
    train_dataset = MaskRcnnDataloader(trainimg, trainlab, trainpre, augmentations=True, area_thd=cfg['data']['area_thd'],
                                       label_is_nos=cfg['data']['label_is_nos'], footprint_mode=cfg['data']['footprint_mode'],
                                       seasons_mode=cfg['data']['seasons_mode'], sar_data=cfg['data']['sar_data'],
                                       if_buffer=cfg['data']['if_buffer'], buffer_storeylevel=cfg['data']['buffer_storeylevel'],
                                       buffer_assist=cfg['data']['buffer_assist'])
    val_dataset = MaskRcnnDataloader(valimg, vallab, valpre, augmentations=False, area_thd=cfg['data']['area_thd'],
                                     label_is_nos=cfg['data']['label_is_nos'], footprint_mode=cfg['data']['footprint_mode'],
                                     seasons_mode=cfg['data']['seasons_mode'], sar_data=cfg['data']['sar_data'],
                                     if_buffer=cfg['data']['if_buffer'], buffer_storeylevel=cfg['data']['buffer_storeylevel'],
                                     buffer_assist=cfg['data']['buffer_assist'])
    traindataloader = torch.utils.data.DataLoader(
        train_dataset,batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)
    evaldataloader = torch.utils.data.DataLoader(
        val_dataset,batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True, collate_fn=val_dataset.collate_fn)

    # Setup Model
    model = get_model(cfg, Maskrcnn=True).to(device)

    # print the model
    start_epoch = 0
    resume = cfg["training"]["resume"]
    print(resume)
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")

    # get all parameters (model parameters + task dependent log variances)
    params = [p for p in model.parameters()]
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # optimizer = torch.optim.Adam(params, lr=cfg["training"]["lr"], betas=(0.9, 0.999))
    optimizer = torch.optim.SGD(params, lr=cfg["training"]["lr"], momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    torch.autograd.set_detect_anomaly(True)
    val_loss_tmp = 10000
    for epoch in range(epochs - start_epoch):
        model.on_epoch_start()
        epoch = start_epoch + epoch
        adjust_learning_rate(optimizer, epoch)
        print('===========================TRAINING================================')
        # train
        model.train()
        train_loss = list()
        loss_classifier, loss_box_reg, loss_nos, loss_mask, loss_objectness, loss_rpn_box_reg = [],[],[],[],[],[]
        step = 0
        pbar = tqdm(traindataloader)
        for imgs, targets in traindataloader:
            step += 1
            imgs = torch.tensor([item.cpu().detach().numpy() for item in imgs]).to(device)
            targets = dicts_to_device(targets, device)
            batch = imgs, targets
            loss_dict = model.forward(batch)
            # print('=================================-------------------------->',loss_dict)
            # loss_classifier.append(loss_dict['loss_classifier'].cpu().detach().numpy())
            # loss_box_reg.append(loss_dict['loss_box_reg'].cpu().detach().numpy())
            loss_nos.append(loss_dict['loss_nos'].cpu().detach().numpy())
            # loss_objectness.append(loss_dict['loss_objectness'].cpu().detach().numpy())
            # loss_mask.append(loss_dict['loss_mask'].cpu().detach().numpy())
            # loss_rpn_box_reg.append(loss_dict['loss_rpn_box_reg'].cpu().detach().numpy())
            loss_values = loss_dict.values()
            loss_values = list(loss_values)
            loss = reduce(add, loss_values)
            train_loss.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update()
            pbar.set_description(f'Epoch:{epoch}, train loss total {np.round(loss.cpu().item(), 3)},'
                                 f' classify {np.round(float(loss_values[0].cpu().detach().numpy()), 3)},'
                                 f' box {np.round(float(loss_values[1].cpu().detach().numpy()), 3)}'
                                 f', nos {np.round(float(loss_values[2].cpu().detach().numpy()), 3)},'
                                 # f' mask {np.round(float(loss_values[3].cpu().detach().numpy()), 3)}'
                                 )

            gc.collect()

        # eval
        print('==========================VALIDATING===============================')
        model.eval()
        val_outs = []
        with torch.no_grad():
            pbar = tqdm(evaldataloader)
            for imgs, targets in evaldataloader:
                pbar.set_description('Validating---')
                pbar.update()
                imgs = torch.tensor([item.cpu().detach().numpy() for item in imgs]).to(device)
                targets = dicts_to_device(targets, device)
                batch = imgs, targets
                out = model.validation_step(batch)
                val_outs.append(out)
            result = model.validation_epoch_end(val_outs)
        val_loss = result['val_loss'].cpu().detach().numpy()
        val_log = result['log']
        print('val loss:', val_loss, '\nlog info :\n', val_log)

        # scheduler step
        # scheduler.step()

        # save models
        # if epoch % 2 == 0: # every five internval
        if epoch == 70:  # every five internval
            savefilename = os.path.join(ckptsavedir, 'finetune_'+str(epoch)+'.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': np.nanmean(train_loss),
                'eval_loss': np.nanmean(val_loss), #*100
                # 'classifier_loss': np.nanmean(loss_classifier),
                # 'loss_box_reg':np.nanmean(loss_box_reg),
                'loss_nos': np.nanmean(loss_nos),
                # 'loss_objectness': np.nanmean(loss_objectness),
                # 'loss_mask': np.nanmean(loss_mask) is loss_mask is not None,
                # 'loss_rpn_box_reg': np.nanmean(loss_rpn_box_reg),
                # 'MAE_all': val_log['MAE_all'].cpu().detach().numpy(),
                # 'F1_all': val_log['F1_all'].cpu().detach().numpy(),
                # 'R_all': val_log['R_all'].cpu().detach().numpy(),
                # 'RMSE_all': val_log['RMSE_all'].cpu().detach().numpy(),
                # 'IoU_NoS_all': val_log['IoU_NoS_all'].cpu().detach().numpy(),
                # 'seg_acc_all': val_log['seg_acc_all'].cpu().detach().numpy(),
                # 'seg_IoU_all': val_log['seg_IoU_all'].cpu().detach().numpy(),

            }, savefilename)
        #

        writer.add_scalar('train loss',
                          (np.nanmean(train_loss)),  # average
                          epoch)
        writer.add_scalar('classifier_loss', np.nanmean(loss_classifier),  # average
                          epoch)
        writer.add_scalar('loss_nos', np.nanmean(loss_nos),  # average
                          epoch)
        # writer.add_scalar('loss_mask', np.nanmean(loss_mask),  # average
        #                   epoch)
        writer.add_scalar('MAE_all', val_log['MAE_all'].cpu().detach().numpy(),  # average
                          epoch)
        # writer.add_scalar('F1_all', val_log['F1_all'].cpu().detach().numpy(),  # average
        #                   epoch)
        writer.add_scalar('res_rel', val_log['res_rel_all'].cpu().detach().numpy(),  # average
                          epoch)
        # writer.add_scalar('R_all', val_log['R_all'].cpu().detach().numpy(),  # average
        #                   epoch)
        writer.add_scalar('RMSE_all', val_log['RMSE_all'].cpu().detach().numpy(),  # average
                          epoch)
        writer.add_scalar('IoU_NoS_all', val_log['IoU_NoS_all'].cpu().detach().numpy(),  # average
                          epoch)
        # writer.add_scalar('MAE', val_log['MAE'].cpu().detach().numpy(),  # average
        #                   epoch)
        # writer.add_scalar('F1', val_log['F1'].cpu().detach().numpy(),  # average
        #                   epoch)
        # writer.add_scalar('P', val_log['P'].cpu().detach().numpy(),  # average
        #                   epoch)
        # writer.add_scalar('R', val_log['R'].cpu().detach().numpy(),  # average
        #                   epoch)
        # writer.add_scalar('res_rel', val_log['res_rel'].cpu().detach().numpy(),  # average
        #                   epoch)
        # writer.add_scalar('seg_R', val_log['seg_R'].cpu().detach().numpy(),  # average
        #                   epoch)
        # writer.add_scalar('RMSE', val_log['RMSE'].cpu().detach().numpy(),  # average
        #                   epoch)
        # writer.add_scalar('IoU_NoS', val_log['IoU_NoS'].cpu().detach().numpy(),  # average
        #                   epoch)
        # writer.add_scalar('seg_P', val_log['seg_P'].cpu().detach().numpy(),  # average
        #                   epoch)
        # writer.add_scalar('seg_acc', val_log['seg_acc'].cpu().detach().numpy(),  # average
        #                   epoch)
        # writer.add_scalar('seg_IoU', val_log['seg_IoU'].cpu().detach().numpy(),  # average
        #                   epoch)
        writer.add_scalar('val loss',
                          val_loss, #average
                          epoch)
        writer.close()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 40:
        lr = cfg["training"]["lr"]
    elif epoch <=100:
        lr = cfg["training"]["lr"] * np.power(0.5, epoch-40)
    elif epoch <=150:
        lr = cfg["training"]["lr"] * np.power(0.5, 40) * np.power(0.1, epoch-100)
    else:
        lr = cfg["training"]["lr"] * np.power(0.5, 40) * np.power(0.1, 50) * np.power(0.05, epoch-150) # 0.0025 before
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr #added



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="../configs/MaskRcnn_res50.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    logdir = os.path.join("../runs", os.path.basename(args.config)[:-4], "V5.1Buffer0Areathd1")
    ckptsavedir = os.path.join('/media/dell/xpl/lsq/CKPTS', "V5.1Buffer0Areathd1")
    make_dir(logdir)
    make_dir(ckptsavedir)
    print(logdir)
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("here begin the log")

    main(cfg, writer, logger, ckptsavedir)
