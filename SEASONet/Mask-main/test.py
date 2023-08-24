'''
用于训练img + season 三分支Unet模型


'''

import yaml
import shutil
import argparse
from tqdm import tqdm
import sys
sys.path.append('/home/dell/lsq/SEASONet_230824')


from torch.utils import data
from models import get_model
from utils import get_logger
from tensorboardX import SummaryWriter #change tensorboardX
from dataloaders.dataloaders import *
from tools import make_dir
from pytorch_tools import *
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # must be the same device with training


def add(num1, num2):
    return num1 + num2


def main(cfg, writer, logger):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))
    # Setup device
    device = torch.device(cfg["training"]["device"])
    # Setup Dataloader
    data_path = cfg["data"]["test_path"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    make_dir(cfg["savepath"])
    buffer_assist = cfg['data']['buffer_assist']
    supervision = cfg['training']['supervision']

    # Load dataset
    if buffer_assist == 'pre_assist':
        trainimg, trainlab, valimg, vallab, _, _, trainpre, valpre, _ = make_dataset(data_path, split=[0, 0, 1],
                                                                                     test_only=False,
                                                                                     buffer_assist=cfg['data']['buffer_assist'],
                                                                                     latnseason=cfg['training']['latnseason'])
    elif supervision != 0:
        _, _, _, _, testimg, testlab, _, _, testsup = make_dataset(data_path, split=[0, 0, 1], test_only=False,
                                                                   latnseason=cfg['training']['latnseason'],
                                                                   sup_name=cfg['data']['sup_name'],
                                                                   supervision=cfg['training']['supervision']
                                                                   )
    else:
        _, _, _, _, testimg, testlab = make_dataset(data_path, split=[0, 0, 1], test_only=False,
                                                   latnseason=cfg['training']['latnseason'],
                                                   supervision=cfg['training']['supervision']
                                                   )
        testpre = None
        testsup = None

    test_dataset = MaskRcnnDataloader(testimg, testlab, lab_sup_path=testsup, augmentations=False, area_thd=cfg['data']['area_thd'],
                                     label_is_nos=cfg['data']['label_is_nos'], footprint_mode=cfg['data']['footprint_mode'],
                                     seasons_mode=cfg['data']['seasons_mode'], sar_data=cfg['data']['sar_data'],
                                     if_buffer=cfg['data']['if_buffer'], buffer_pixels=cfg['data']['buffer_pixels'],
                                     buffer_assist=cfg['data']['buffer_assist'], latnseason=cfg['training']['latnseason'],
                                     supervision=cfg['training']['supervision']
                                      )

    testdataloader = torch.utils.data.DataLoader(test_dataset,
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=test_dataset.collate_fn)

    # Setup Model
    model = get_model(cfg, Maskrcnn=True).to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # print the model
    resume = cfg["test"]["resume"]
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")

    model.eval()
    test_outs = []
    with torch.no_grad():
        # for img, target, sups in tqdm(testdataloader):
        for img, target in tqdm(testdataloader):
            img = torch.tensor([item.cpu().detach().numpy() for item in img]).cuda()
            target = dicts_to_device(target, device)
            batch = img, target
            out = model.test_step(batch, save_fig=True)
            # save_predict(out, target, cfg["data"]["img_rows"], cfg["savepath"])
            test_outs.append(out)
        result = model.test_epoch_end(test_outs)
    val_loss = result['val_loss'].cpu().detach().numpy()
    val_log = result['log']
    print('val loss:', val_loss, '\nlog info :\n', val_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="../configs/MaskRcnn_res50.yml",
        # default="../configs/SEASONet_sup.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    logdir = os.path.join("../runs", os.path.basename(args.config)[:-4], "TEST_20230824")
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("here begin the log")

    main(cfg, writer, logger)
