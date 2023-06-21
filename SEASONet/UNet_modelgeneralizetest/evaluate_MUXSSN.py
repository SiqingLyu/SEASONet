'''
test ningbo images
'''
import os
import yaml
import shutil
import torch
import random
import argparse
import numpy as np
from skimage.measure import label, regionprops
import sys
sys.path.append('/home/dell/lsq/LSQNetModel_on')
# sys.path.append('/media/dell/shihaoze/lsq/LSQNetModel/Unet-main')

from models import get_model
from utils import get_logger
from tensorboardX import SummaryWriter
from dataloaders.dataloaders import *
from Data.LabelTargetProcessor import LabelTarget
# import sklearn.metrics
import matplotlib.pyplot as plt
import tifffile as tif
os.environ['CUDA_VISIBLE_DEVICES']='1'


def process_nan(imgdata):
    imgdata = np.nan_to_num(imgdata)
    imgdata[imgdata < (-3.4028235e+10)] = 0
    return imgdata


def normalize_maxmin(array, path = 'no file'):
    '''
    Normalize the array
    '''
    array = np.nan_to_num(array)
    array[(array >= 10000) | (array <= -10000)] = 0
    # array[array < (-3.4028235e+10)] = 0
    # array[array > (10000)] = 0
    mx = np.max(array)
    mn = np.min(array)

    if mx == mn:
        print('All-slice same data =======================>'+path)
    if mx >= 10000:
        print('data overflow =======================>'+path)
    if mn <= -10000:
        print('data overflow =======================>'+path)
    assert mx < 10000
    assert mn > -10000

    assert (mx != mn)
    t = (array-mn)/(mx-mn)
    return t


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def main(cfg, writer, logger):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))
    # Setup device
    device = torch.device(cfg["training"]["device"])
    # Setup Dataloader
    data_path =  cfg["data"]["path"]
    test_path =  cfg["data"]["test_path"]
    n_classes = cfg["data"]["n_class"]
    n_maxdisp = cfg["data"]["n_maxdisp"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    learning_rate = cfg["training"]["learning_rate"]
    patchsize = cfg["data"]["img_rows"]
    seasons_mode = cfg["data"]["seasons_mode"]
    center_point = cfg["test"]["CenterPoint"]

    _, _, _, _, testimg, testlab = make_dataset(test_path, split=[0,0,1], test_only=True)

    model = get_model(cfg["model"], n_maxdisp=n_maxdisp, n_classes=n_classes).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    resume = cfg["test"]["resume"]
    # resume = r'/media/dell/shihaoze/lsq/LSQNetModel/runs/ALLvvhnetu_S2_withfootprint/V1/finetune_230.tar'
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'],False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")

    make_dir(cfg["savepath"])
    model.eval()
    all_rmse = 0.
    for idx, imgpath in enumerate(testimg):
        name = os.path.basename(testlab[idx])
        y_true = tif.imread(testlab[idx])
        y_true = y_true.astype(np.int16)
        # random crop: test and train is the same
        mux_img = Images(img_path=imgpath[0])
        mux_img.normalize()
        img = Images(img_data=mux_img.img_data)
        if seasons_mode == 'intact':
            spr_img = Images(img_path=imgpath[3])
            spr_img.normalize()
            smr_img = Images(img_path=imgpath[4])
            smr_img.normalize()
            fal_img = Images(img_path=imgpath[5])
            fal_img.normalize()
            wnt_img = Images(img_path=imgpath[6])
            wnt_img.normalize()
            cat_data_list = [spr_img.img_data, smr_img.img_data, fal_img.img_data, wnt_img.img_data]
            img.cat_images(cat_datas=cat_data_list)

        img = img.img_data
        img[img > 1] = 1  # ensure data range is 0-1
        img = np.nan_to_num(img)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()
        with torch.no_grad():
            y_res = model(img.to(device))
            y_pred = y_res[0] # height
            y_pred = y_pred.cpu().detach().numpy()
            y_pred = np.squeeze(y_pred)
            y_seg = y_res[1]  # seg
            y_seg = y_seg.cpu().detach().numpy()
            y_seg = np.argmax(y_seg.squeeze(), axis=0)  # C H W=>  H W
            precision, recall, f1score = metricsperclass(y_true, y_seg, value=1) #
            y_predmasked = np.where(y_true > 0, y_pred, 0.)
            rmse = myrmse(y_true, y_predmasked)
        print("filename: ", name, "RMSE:", rmse)
        print("precision: ", precision, " recall:", recall, " F1:", f1score)
        all_rmse += rmse
        make_dir(os.path.join(cfg["savepath"], 'lab'))
        make_dir(os.path.join(cfg["savepath"], 'pred'))
        make_dir(os.path.join(cfg["savepath"], 'seg'))

        if center_point:
            y_pred = get_preds_center_ponint(y_true, y_predmasked)

        tif.imsave((os.path.join(cfg["savepath"], 'lab', name)), y_true)
        tif.imsave((os.path.join(cfg["savepath"], 'pred', name)), y_pred)
        tif.imsave((os.path.join(cfg["savepath"], 'seg', name)), y_seg.astype(np.uint8))
    print("mean RMSE:", all_rmse / len(testimg))


def get_preds_center_ponint(y_true, y_pred, storey_thd=7):
    y_pred[y_pred < 0] = 0
    Tar = LabelTarget(label_data=y_true).to_target_cpu(background=0, area_thd=1)
    masks = Tar['masks']
    areas = Tar['area']
    for ii in range(len(masks)):
        mask = masks[ii]
        area = areas[ii]
        y_pred_temp = np.where(mask == 1, y_pred, 0)
        storey_thd_temp = storey_thd
        if area < 100:
            y_pred[mask == 1] = np.max(y_pred_temp).astype('int64')
        else:
            y_pred_outlined = np.where(y_pred_temp > storey_thd_temp, y_pred, 0).astype('int64')
            while np.all(y_pred_outlined == 0):
                storey_thd_temp -= 1
                y_pred_outlined = np.where(y_pred_temp > storey_thd_temp, y_pred, 0).astype('int64')
            assert storey_thd_temp > 0
            y_pred_other = np.where(y_pred_temp <= storey_thd_temp, y_pred, 0).astype('int64')
            new_masks_outlined = LabelTarget(label_data=y_pred_outlined).to_target_cpu(background=0, area_thd=1)['masks']
            for new_mask in new_masks_outlined:
                new_pred = np.max(y_pred_outlined[new_mask==1])
                y_pred[new_mask==1] = new_pred
            new_masks_other = LabelTarget(label_data=y_pred_other).to_target_cpu(background=0, area_thd=1)['masks']
            for new_mask in new_masks_other:
                new_pred = np.max(y_pred_other[new_mask==1])
                y_pred[new_mask==1] = new_pred
    return y_pred.astype('uint8')


def metricsperclass(y_true, y_pred, value):
    y_pred = y_pred.flatten()
    y_true = np.where(y_true>0, np.ones_like(y_true), np.zeros_like(y_true)).flatten()

    tp=len(np.where((y_true==value) & (y_pred==value))[0])
    tn=len(np.where(y_true==value)[0])
    fn = len(np.where(y_pred == value)[0])
    precision = tp/(1e-10+fn)
    recall = tp/(1e-10+tn)
    f1score = 2*precision*recall/(precision+recall+1e-10)
    return precision, recall, f1score


def myrmse(y_true, ypred):
    diff=y_true.flatten()-ypred.flatten()

    return np.sqrt(np.mean(diff*diff))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="../configs/M3Net_SSN_TESTCenterPoint.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    #run_id = random.randint(1, 100000)
    logdir = os.path.join("../runs", os.path.basename(args.config)[:-4], "V1")
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    main(cfg, writer, logger)
