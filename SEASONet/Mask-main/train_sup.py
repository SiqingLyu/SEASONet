# -*- coding: UTF-8 -*-
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
sys.path.append('/home/dell/lsq/LSQNetModel_generalize')
# sys.path.append('/media/dell/shihaoze/lsq/LSQNetModel/Mask-main')
# sys.setrecursionlimit(1000)
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
from pympler import tracker,summary,muppy
import tracemalloc
from memory_profiler import profile
import torch.nn.functional as F
# from segmentation_models_pytorch_revised import DeepLabV3Plus
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

torch.set_num_threads(3)

def add(num1, num2):
    return num1 + num2

def detach_all(contain):
    if isinstance(contain, dict):
        ret = {}
        for k, v in contain.items():
            if hasattr(v, 'grad_fn') and v.grad_fn:
                ret[k] = v.detach()
            else:
                ret[k] = v
    return ret

def dict_to_arr(dict_):
    result = dict_.items()  # 此处可根据情况选择 .values or .keys
    data = list(result)
    numpyArray = np.array(data)
    return numpyArray



# async def print_on_response(request, response):
#     gc.collect()
#     snapshot1 = tracemalloc.take_snapshot()
#
#     print("[ Top 10 differences ]")
#     for stat in top_stats[:10]:
#         if stat.size_diff < 0:
#             continue
#         print(stat)
#     snapshot = tracemalloc.take_snapshot()


# @profile(precision=4, stream=open("memory_profiler.log", "w+"))  # 统计内存的精度
def main(cfg, writer, logger, ckptsavedir):

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
    supervision = cfg['training']['supervision']
    # Load dataset
    if buffer_assist == 'pre_assist':
        trainimg, trainlab, valimg, vallab, _, _, trainpre, valpre, _ = make_dataset(data_path, split=[0.8, 0.2, 0.0],
                                                                                     test_only=False,
                                                                                     buffer_assist=cfg['data']['buffer_assist'],
                                                                                     latnseason=cfg['training']['latnseason'])
    elif supervision:
        trainimg, trainlab, valimg, vallab, _, _, trainsup, valsup, _ = make_dataset(data_path, split=[0.8, 0.2, 0.0],
                                                                                     test_only=False,
                                                                                     supervision=cfg['training']['supervision'],
                                                                                     latnseason=cfg['training']['latnseason'])
    else:
        trainimg, trainlab, valimg, vallab, _, _ = make_dataset(data_path, split=[0.8, 0.2, 0.0], test_only=False,
                                                                latnseason=cfg['training']['latnseason'])
        trainpre = None
        valpre = None
        trainsup= None
        valsup = None
    train_dataset = MaskRcnnDataloader(trainimg, trainlab, lab_sup_path=trainsup, augmentations=True, area_thd=cfg['data']['area_thd'],
                                       label_is_nos=cfg['data']['label_is_nos'], footprint_mode=cfg['data']['footprint_mode'],
                                       seasons_mode=cfg['data']['seasons_mode'], sar_data=cfg['data']['sar_data'],
                                       if_buffer=cfg['data']['if_buffer'], buffer_pixels=cfg['data']['buffer_pixels'],
                                       buffer_assist=cfg['data']['buffer_assist'], latnseason=cfg['training']['latnseason'],
                                       supervision=cfg['training']['supervision'])
    val_dataset = MaskRcnnDataloader(valimg, vallab, lab_sup_path=valsup, augmentations=False, area_thd=cfg['data']['area_thd'],
                                     label_is_nos=cfg['data']['label_is_nos'], footprint_mode=cfg['data']['footprint_mode'],
                                     seasons_mode=cfg['data']['seasons_mode'], sar_data=cfg['data']['sar_data'],
                                     if_buffer=cfg['data']['if_buffer'], buffer_pixels=cfg['data']['buffer_pixels'],
                                     buffer_assist=cfg['data']['buffer_assist'], latnseason=cfg['training']['latnseason'],
                                     supervision=False)
    traindataloader = torch.utils.data.DataLoader(
        train_dataset,batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, collate_fn=train_dataset.collate_fn)
    evaldataloader = torch.utils.data.DataLoader(
        val_dataset,batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, collate_fn=val_dataset.collate_fn)

    # Setup Model
    model = get_model(cfg, Maskrcnn=True).to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

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
        loss_sup = []
        step = 0
        pbar = tqdm(traindataloader)
        for imgs, targets, sups in traindataloader:
            step += 1
            imgs = torch.tensor([item.cpu().detach().numpy() for item in imgs]).to(device)
            sups = torch.tensor([item.cpu().detach().numpy() for item in sups]).to(device)
            targets = dicts_to_device(targets, device)

            batch = imgs, targets, sups

            loss_dict = model.forward(batch)
            loss_nos.append(loss_dict['loss_nos'].cpu().detach().numpy())
            loss_sup.append(loss_dict['agent_loss'].cpu().detach().numpy())

            # print(loss_dict.keys())
            loss_values = loss_dict.values()
            loss_values = list(loss_values)
            loss = reduce(add, loss_values)

            train_loss.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update()
            pbar.set_description(f'Epoch:{epoch}, train loss total {np.round(loss.cpu().item(), 3)},'
                                 # f' classify {np.round(float(loss_values[0].cpu().detach().numpy()), 3)},'
                                 # f' box {np.round(float(loss_values[1].cpu().detach().numpy()), 3)}'
                                 f', nos {np.round(float(loss_values[0].cpu().detach().numpy()), 3)},'
                                 f', agent {np.round(float(loss_values[1].cpu().detach().numpy()), 3)},'
                                 # f' mask {np.round(float(loss_values[3].cpu().detach().numpy()), 3)}'
                                 )

            gc.collect()
            # print('train loss: ', loss.item())

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
                # sups = torch.tensor([item.cpu().detach().numpy() for item in sups]).to(device)
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
        if epoch % 20 == 0: # every five internval
        # if epoch == 70:  # every five internval
            savefilename = os.path.join(ckptsavedir, 'finetune_'+str(epoch)+'.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': np.nanmean(train_loss),
                'eval_loss': np.nanmean(val_loss), #*100
                # 'classifier_loss': np.nanmean(loss_classifier),
                # 'loss_box_reg':np.nanmean(loss_box_reg),
                'loss_nos': np.nanmean(loss_nos),
                'loss_agent': np.nanmean(loss_sup),
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
        # writer.add_scalar('classifier_loss', np.nanmean(loss_classifier),  # average
        #                   epoch)
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
    if epoch <= 25:
        lr = cfg["training"]["lr"]
    # elif epoch <= 30:
    #     lr = cfg["training"]["lr"] * np.power(0.5, 1)
    # elif epoch <= 40:
    #     lr = cfg["training"]["lr"] * np.power(0.5, 2)
    elif epoch <=40:
        lr = cfg["training"]["lr"] * np.power(0.5, epoch-25)
    elif epoch <=100:
        lr = cfg["training"]["lr"] * np.power(0.5, 15) * np.power(0.1, epoch-40)
    else:
        lr = cfg["training"]["lr"] * np.power(0.5, 15) * np.power(0.1, 60) * np.power(0.05, epoch-100) # 0.0025 before
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
        default="../configs/SEASONet_sup.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    logdir = os.path.join("../runs", os.path.basename(args.config)[:-4], "SEASONet_SUP_decayat25_lat")
    ckptsavedir = os.path.join('/media/dell/徐朋磊的移动固态硬盘/lsq/CKPTS', "SEASONet_SUP_decayat25_lat")
    make_dir(logdir)
    make_dir(ckptsavedir)
    print(logdir)
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("here begin the log")

    main(cfg, writer, logger, ckptsavedir)
