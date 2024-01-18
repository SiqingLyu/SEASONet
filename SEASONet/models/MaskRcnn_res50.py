#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch

import models.detection as MaskrcnnModels
from torchvision.ops import MultiScaleRoIAlign
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from pytorch_tools import *
import copy
from torch.nn.functional import l1_loss, mse_loss
from Data.LabelTargetProcessor import LabelTarget
from tools import make_dir

import tracemalloc
BACKGROUND = 0


class nos_head(nn.Module):
    def __init__(self, in_channels, representation_size):
        super(nos_head, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class nos_predictor(nn.Module):
    def __init__(self, in_channels):
        super(nos_predictor, self).__init__()

        self.nos_predict_end = nn.Linear(in_channels, 1)

    def forward(self, x, nos_proposals):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        nos = self.nos_predict_end(x)
        boxes_per_image = [len(l) for l in nos_proposals]
        nos_list = nos.split(boxes_per_image, dim=0)
        return nos_list


def nos_loss(self, gt_nos, nos_predict, fname):
    nos_list = [self.metadata_block(n, f) for n, f in zip(nos_predict, fname)]
    gt_nos = torch.cat(gt_nos)
    pre_nos = torch.cat(nos_list)
    has_nos = gt_nos != BACKGROUND
    assert has_nos.any()
    preds = pre_nos.squeeze()[has_nos]
    gts = gt_nos[has_nos]
    res_rel = torch.mean(torch.abs(preds - gts) / (gts+0.0000000001))
    storey_loss = F.smooth_l1_loss(preds,
                                   gts,
                                   reduction="sum")
    storey_loss = storey_loss / (has_nos.sum() + 0.00000000001)
    storey_loss = res_rel + storey_loss
    return storey_loss


def metadata_block(self, predict, fname):
    return predict


class MaskRcnn_res50(nn.Module):
    def __init__(self, params):
        super(MaskRcnn_res50, self).__init__()
        self.params = params
        self.num_classes = params['data']['num_classes']
        self.exp_folder = params['savepath']
        self.data_path = params['data']['path']
        self.test_data_path = params['data']['test_path']
        self.img_size = params['data']['img_rows']
        self.gt_rpn_training = params['training']['gt_rpn_training']
        assert self.gt_rpn_training, "this code doesn't support the situation of gt_tpn_training is False"
        self.input_channels = params['data']['channels']
        self.set_model()
        self.init_end()

    def forward(self, x):
        return self.main_model(*x)

    def set_model(self):
        self.main_model = MaskrcnnModels.maskrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=False, mask_branch=True)
        # self.main_model.backbone.body.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 层数预测分支
        self.main_model.roi_heads.nos_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)
        representation_size = 1024
        resolution = self.main_model.roi_heads.nos_roi_pool.output_size[0]
        self.main_model.roi_heads.nos_head = nos_head(self.main_model.backbone.out_channels * resolution ** 2,
                                                      representation_size)
        self.main_model.roi_heads.nos_predictor = nos_predictor(representation_size)
        self.main_model.roi_heads.nos_loss = MethodType(nos_loss, self.main_model.roi_heads)
        self.main_model.roi_heads.metadata_block = MethodType(metadata_block, self.main_model.roi_heads)

        self.main_model.roi_heads.box_roi_pool = None
        self.main_model.roi_heads.box_head = None
        self.main_model.roi_heads.box_predictor = None
        self.main_model.roi_heads.mask_roi_pool = None
        self.main_model.roi_heads.mask_head = None
        self.main_model.roi_heads.mask_predictor = None

    def init_end(self):
        if not hasattr(self, 'gt_rpn_training'):
            self.gt_rpn_training = False

    def set_gt_rpn(self, book):
        if book:
            self.main_model.rpn.gt_rpn = True
            self.main_model.roi_heads.gt_rpn = True
            self.main_model.roi_heads.fg_bg_sampler.positive_fraction = 1.0
        else:
            self.main_model.rpn.gt_rpn = False
            self.main_model.roi_heads.gt_rpn = False
            self.main_model.roi_heads.fg_bg_sampler.positive_fraction = self.positive_fraction_bak

    def on_epoch_start(self):
        self.set_gt_rpn(self.gt_rpn_training)

    def postprocess_output(self, output, gts):
        return output, copy.deepcopy(gts)

    def validation_step(self, batch, save_opt=False):
        if self.gt_rpn_training:

            self.set_gt_rpn(True)

            images, gts = batch

            output = self([images, copy.deepcopy(gts)])
            output, gts_post = self.postprocess_output(output, gts)
            nos_preds, nos_gts = torch.tensor([]), torch.tensor([])
            for inx, (img, target, predict) in enumerate(zip(images, gts_post, output)):
                predict = detach_all(predict)
                if save_opt:
                    self.save_predict(predict, target, self.exp_folder)

                nos_pred = predict['nos'].cpu()
                nos_gt = target['nos'].float().cpu()

                nos_preds = torch.cat((nos_preds, nos_pred))
                nos_gts = torch.cat((nos_gts, nos_gt))

            return nos_preds, nos_gts
        else:
            return

    def get_val_result(self, nos_preds, nos_gts):
        log_dict = {}
        MAE_all, RMSE_all, res_rel_all, IoU_NoS_all = NoS_metric(nos_preds, nos_gts)
        log_dict_all = make_log('', locals(), MAE_all, RMSE_all, res_rel_all, IoU_NoS_all)
        val_loss = MAE_all
        log_dict.update(log_dict_all)

        return {
            'val_loss': val_loss,
            'log': log_dict
        }

    def validation_epoch_end(self, preds, gts):
        out = self.get_val_result(preds, gts)
        return out

    def test_epoch_end(self, preds, gts):
        out = self.validation_epoch_end(preds, gts)
        return out

    def test_step(self, batch, save_fig=True):
        return self.validation_step(batch, save_fig)

    def save_predict(self, predict, target, save_path, opt='v2'):
        file_name = target['file_name']
        pred_save_path = make_dir(os.path.join(save_path, 'pred'))
        # lab_save_path = os.path.join(make_dir(os.path.join(save_path, 'lab')), file_name +'.tif')
        # lab_data = tif.imread(os.path.join(self.test_data_path, 'label', file_name+'.tif'))
        # tif.imsave(lab_save_path, lab_data)
        img_path = self.test_data_path + '/image/optical/' + file_name + '.tif'
        Target = LabelTarget(target_data=target)
        Target.save_targetdraw_image(pred_save_path, file_name, opt=opt, predicts=predict, img_path=img_path, if_round=False)


def NoS_metric(TP_p, TP_t):
    '''return MAE, RMSE, res_rel, IoU_NoS'''

    if len(TP_t) == 0:
        MAE, RMSE, res_rel, IoU_NoS = torch.tensor([9999] * 4)
        return MAE, RMSE, res_rel, IoU_NoS
    if isinstance(TP_t, np.ndarray):
        TP_t, TP_p = torch.from_numpy(TP_t), torch.from_numpy(TP_p)
    valid_gt = TP_t.cuda()
    # print(valid_pre.device, valid_gt.device)
    valid_pre = TP_p.cuda()
    MAE = l1_loss(valid_pre, valid_gt)
    RMSE = mse_loss(valid_pre, valid_gt) ** 0.5

    res_rel = torch.abs(valid_gt - valid_pre) / (valid_gt + 0.000001)
    res_rel = res_rel.mean()
    nos_stack = torch.stack([valid_gt.type_as(valid_pre), valid_pre])
    nos_min, _ = nos_stack.min(0)
    nos_max, _ = nos_stack.max(0)
    IoU_NoS = nos_min / (nos_max + 0.0000001)
    IoU_NoS = IoU_NoS.mean()
    return MAE, RMSE, res_rel, IoU_NoS
