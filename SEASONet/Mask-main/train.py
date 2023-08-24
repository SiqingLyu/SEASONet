# -*- coding: UTF-8 -*-
import yaml
import shutil
import random
import argparse
from tqdm import tqdm
from functools import reduce
import sys
sys.path.append('/home/dell/lsq/SEASONet_230824')

from torch.utils import data
from models import get_model
from utils import get_logger
from tensorboardX import SummaryWriter #change tensorboardX
from dataloaders.dataloaders import *
from tools import *
from pytorch_tools import *
from thop import profile
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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
    shuffle = cfg['training']['shuffle']

    # Load dataset

    trainimg, trainlab, valimg, vallab, _, _ = make_dataset(data_path, split=[0.8, 0.2, 0.0], test_only=False)

    train_dataset = MaskRcnnDataloader(trainimg, trainlab,augmentations=True, area_thd=cfg['data']['area_thd'],
                                       label_is_nos=cfg['data']['label_is_nos'], seasons_mode=cfg['data']['seasons_mode'],
                                       if_buffer=cfg['data']['if_buffer'], buffer_pixels=cfg['data']['buffer_pixels'])
    val_dataset = MaskRcnnDataloader(valimg, vallab, augmentations=False, area_thd=cfg['data']['area_thd'],
                                     label_is_nos=cfg['data']['label_is_nos'], seasons_mode=cfg['data']['seasons_mode'],
                                     if_buffer=cfg['data']['if_buffer'], buffer_pixels=cfg['data']['buffer_pixels'])
    traindataloader = torch.utils.data.DataLoader(
        train_dataset,batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, collate_fn=train_dataset.collate_fn)
    evaldataloader = torch.utils.data.DataLoader(
        val_dataset,batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, collate_fn=val_dataset.collate_fn)

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
    for epoch in range(epochs - start_epoch):
        model.on_epoch_start()
        epoch = start_epoch + epoch
        adjust_learning_rate(optimizer, epoch)
        print('===========================TRAINING================================')
        # train
        model.train()
        train_loss = list()
        loss_nos = []
        step = 0
        pbar = tqdm(traindataloader)
        for imgs, targets in traindataloader:
            step += 1
            imgs = torch.tensor([item.cpu().detach().numpy() for item in imgs]).to(device)
            targets = dicts_to_device(targets, device)
            batch = imgs, targets

            loss_dict = model.forward(batch)

            # print(loss_dict)
            # macs, params = profile(model, inputs=(batch,))
            # print(f"==============PROFILE============\nFLOPs:{macs}, params:{params}\n\n")

            loss_nos.append(loss_dict['loss_nos'].cpu().detach().numpy())

            loss_values = loss_dict.values()
            loss_values = list(loss_values)

            loss = reduce(add, loss_values)

            train_loss.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update()
            pbar.set_description(f'Epoch:{epoch}, train loss total {np.round(loss.cpu().item(), 3)},'
                                 f', nos {np.round(float(loss_values[0].cpu().detach().numpy()), 3)},'
                                 )

            gc.collect()
        # eval
        print('==========================VALIDATING===============================')
        model.eval()
        val_preds = torch.tensor([])
        val_gts = torch.tensor([])
        with torch.no_grad():
            pbar = tqdm(evaldataloader)
            for imgs, targets in evaldataloader:
                pbar.set_description('Validating---')
                pbar.update()
                imgs = torch.tensor([item.cpu().detach().numpy() for item in imgs]).to(device)
                targets = dicts_to_device(targets, device)
                batch = imgs, targets
                nos_preds, nos_gts = model.validation_step(batch)
                val_preds = torch.cat((val_preds, nos_preds))
                val_gts = torch.cat((val_gts, nos_gts))

            assert len(val_gts) == len(val_gts), "pred and gt should have the same length"
            # print(val_gts, val_preds)
            result = model.validation_epoch_end(val_preds, val_gts)
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
                'loss_nos': np.nanmean(loss_nos),
                'MAE_all': val_log['MAE_all'].cpu().detach().numpy(),
                'RMSE_all': val_log['RMSE_all'].cpu().detach().numpy()

            }, savefilename)
        #

        writer.add_scalar('train loss',
                          (np.nanmean(train_loss)),  # average
                          epoch)
        writer.add_scalar('loss_nos', np.nanmean(loss_nos),  # average
                          epoch)
        writer.add_scalar('MAE_all', val_log['MAE_all'].cpu().detach().numpy(),  # average
                          epoch)

        writer.add_scalar('RMSE_all', val_log['RMSE_all'].cpu().detach().numpy(),  # average
                          epoch)

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
        lr = cfg["training"]["lr"] * np.power(0.5, 60) * np.power(0.1, epoch-100)
    else:
        lr = cfg["training"]["lr"] * np.power(0.5, 60) * np.power(0.1, 50) * np.power(0.05, epoch-150) # 0.0025 before
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
        default="../configs/SEASONet.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    logdir = os.path.join("../runs", os.path.basename(args.config)[:-4], "SEASONet_Nomaskboxbranch")
    ckptsavedir = os.path.join('/media/dell/xpl/lsq/CKPTS', "SEASONet_Nomaskboxbranch")
    make_dir(logdir)
    make_dir(ckptsavedir)
    print(logdir)
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("here begin the log")

    main(cfg, writer, logger, ckptsavedir)
