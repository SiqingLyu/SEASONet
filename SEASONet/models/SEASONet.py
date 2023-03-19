#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
from models.submodule import *
import models.detection as MaskrcnnModels
from models.detection.faster_rcnn import FastRCNNPredictor
from models.detection.mask_rcnn import MaskRCNNPredictor
from models.detection.resnet import Bottleneck
from torchvision.ops import MultiScaleRoIAlign
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from pytorch_tools import *
import copy
from torchvision.ops import boxes as box_ops
from torch.nn.functional import l1_loss, mse_loss
import tifffile as tif
from Data.LabelTargetProcessor import LabelTarget
from tools import make_dir
from typing import Type, Any, Callable, Union, List, Optional
import tracemalloc
BACKGROUND = 0
sys.path.append('/SEASONet')


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
        self.box_nms_thresh = params['model']['box_nms_thresh']
        self.box_score_thresh = params['model']['box_score_thresh']
        self.num_classes = params['data']['num_classes']
        self.exp_folder = params['savepath']
        self.data_path = params['data']['path']
        self.test_data_path = params['data']['test_path']
        self.img_size = params['data']['img_rows']
        self.gt_rpn_training = params['training']['gt_rpn_training']
        self.input_channels = params['data']['channels']
        self.set_model()
        self.init_end()

    def forward(self, x):
        return self.main_model(*x)

    def set_model(self):
        faster_rcnn_kargs = {
            'box_nms_thresh': self.box_nms_thresh,
            'box_score_thresh': self.box_score_thresh
            # 'box_fg_iou_thresh':0.7,
            # 'box_bg_iou_thresh':0.7
        }
        self.main_model = MaskrcnnModels.maskrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=False, mask_branch=True, **faster_rcnn_kargs)
        self.main_model.backbone.body.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # # Box branch
        in_features = self.main_model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.main_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # Mask branch
        # now get the number of input features for the mask classifier
        in_features_mask = self.main_model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = self.main_model.roi_heads.mask_predictor.conv5_mask.out_channels
        # and replace the mask predictor with a new one
        self.main_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                                     hidden_layer,
                                                                     self.num_classes)

        # Nos head
        self.main_model.roi_heads.nos_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)
        representation_size = 1024
        resolution = self.main_model.roi_heads.nos_roi_pool.output_size[0]
        print('----------------------------->', self.main_model.backbone.out_channels * resolution ** 2, self.main_model.backbone.out_channels )
        self.main_model.roi_heads.nos_head = nos_head(self.main_model.backbone.out_channels * resolution ** 2,
                                                      representation_size)
        self.main_model.roi_heads.nos_predictor = nos_predictor(representation_size)
        self.main_model.roi_heads.nos_loss = MethodType(nos_loss, self.main_model.roi_heads)
        self.main_model.roi_heads.metadata_block = MethodType(metadata_block, self.main_model.roi_heads)

    def init_end(self):
        self.iou_threshold = 0.5
        if not hasattr(self, 'loss_weight'):
            self.loss_weight = {}
        if not hasattr(self, 'gt_rpn_training'):
            self.gt_rpn_training = False
        self.positive_fraction_bak = self.main_model.roi_heads.fg_bg_sampler.positive_fraction
        self.reset_metric_collection()

    def reset_metric_collection(self):
        self.nos_collection = result_collection('nos')
        self.area_collection = result_collection('area')
        self.score_collection = result_collection('scores')
        self.iou_collection = result_collection('iou')
        self.file_names = []
        self.seg_metirx = [[] for i in range(5)]
        self.nos_collection_gt = result_collection('nos')
        self.score_collection_gt = result_collection('scores')
        self.seg_metirx_gt = [[] for i in range(5)]

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
        images, gts = batch
        show_inxes = [np.random.randint(len(images))]

        if not self.gt_rpn_training:
            self.set_gt_rpn(False)
            output = self([images, copy.deepcopy(gts)])
            output, gts_post = self.postprocess_output(output, gts)
            for inx, (img, target, predict) in enumerate(zip(images, gts_post, output)):
                predict = detach_all(predict)
                if save_opt:
                    self.save_predict(predict, target, self.exp_folder)
                t_boxes = target['boxes']
                target['labels'] = target['labels'].float()
                p_boxes = predict['boxes']
                p_score = predict['scores']
                if len(p_boxes) == 0:
                    TP_pre_inx, TP_gt_inx, FP_p_inx = [[]] * 3
                    FN_t_inx = list(range(len(t_boxes)))
                    iou_pre = p_score
                else:
                    match_quality_matrix = box_ops.box_iou(t_boxes, p_boxes)
                    TP_pre_inx, TP_gt_inx, FP_p_inx, FN_t_inx, iou_pre = get_match_inx(match_quality_matrix,
                                                                                       self.iou_threshold)
                predict['iou'] = iou_pre
                TP_p, TP_t, FP_p, FN_t = self.nos_collection.collect(target, predict, TP_pre_inx, TP_gt_inx, FP_p_inx,
                                                                     FN_t_inx)
                _ = self.area_collection.collect(target, None, TP_pre_inx, TP_gt_inx, FP_p_inx, FN_t_inx)
                _ = self.score_collection.collect(None, predict, TP_pre_inx, TP_gt_inx, FP_p_inx, FN_t_inx)
                _ = self.iou_collection.collect(None, predict, TP_pre_inx, TP_gt_inx, FP_p_inx, FN_t_inx)
                # self.file_names.append(target["file_name"])

                if 'masks' in predict.keys():

                    mask_true = target['masks'].any(0).bool().cuda()
                    mask_pred = (predict['masks'] > 0.5).any(0)[0].cuda()
                    inter = (mask_true & mask_pred).sum()
                    union = (mask_true | mask_pred).sum()
                    TPTN = (mask_true == mask_pred).sum()
                    pos, gt = mask_pred.sum(), mask_true.sum()
                    self.seg_metirx[0].append(inter)
                    self.seg_metirx[1].append(union)
                    self.seg_metirx[2].append(TPTN)
                    self.seg_metirx[3].append(pos)
                    self.seg_metirx[4].append(gt)
                if inx in show_inxes:
                    self.sample_step_end(locals())
        else:
            self.set_gt_rpn(True)
            output = self([images, copy.deepcopy(gts)])
            output, gts_post = self.postprocess_output(output, gts)
            for inx, (img, target, predict) in enumerate(zip(images, gts_post, output)):
                predict = detach_all(predict)
                if save_opt:
                    self.save_predict(predict, target, self.exp_folder)
                t_boxes = target['boxes']
                target['labels'] = target['labels'].float()
                p_boxes = predict['boxes']
                p_score = predict['scores']
                if len(p_boxes) == 0:
                    TP_pre_inx, TP_gt_inx, FP_p_inx = [[]] * 3
                    FN_t_inx = list(range(len(t_boxes)))
                else:
                    match_quality_matrix = box_ops.box_iou(t_boxes, p_boxes)
                    TP_pre_inx, TP_gt_inx, FP_p_inx, FN_t_inx, iou_pre = get_match_inx(match_quality_matrix, 0.7)
                TP_p, TP_t, FP_p, FN_t = self.nos_collection_gt.collect(target, predict, TP_pre_inx, TP_gt_inx,
                                                                        FP_p_inx,
                                                                        FN_t_inx)
                # _ = self.area_collection.collect(target, None, TP_pre_inx, TP_gt_inx, FP_p_inx, FN_t_inx)
                _ = self.score_collection_gt.collect(None, predict, TP_pre_inx, TP_gt_inx, FP_p_inx, FN_t_inx)
                self.file_names.append(target["file_name"])

                if 'masks' in predict.keys():

                    mask_true = target['masks'].any(0).bool().cuda()
                    mask_pred = (predict['masks'] > 0.5).any(0)[0].cuda()
                    inter = (mask_true & mask_pred).sum()
                    union = (mask_true | mask_pred).sum()
                    TPTN = (mask_true == mask_pred).sum()
                    pos, gt = mask_pred.sum(), mask_true.sum()
                    self.seg_metirx_gt[0].append(inter)
                    self.seg_metirx_gt[1].append(union)
                    self.seg_metirx_gt[2].append(TPTN)
                    self.seg_metirx_gt[3].append(pos)
                    self.seg_metirx_gt[4].append(gt)
                if inx in show_inxes:
                    self.sample_step_end(locals())

        return {}

    def get_val_result(self, output):
        log_dict = {}
        if not self.gt_rpn_training:
            TP_p, TP_t, FP_p, FN_t = self.nos_collection.cat_all()
            P, R, F1 = PRF1(TP_p, TP_t, FP_p, FN_t)
            if self.params['training']['nos_task']:
                MAE, RMSE, res_rel, IoU_NoS = NoS_metric(TP_p, TP_t, FP_p, FN_t)
                log_dict = make_log('', locals(), MAE, F1, P, R, RMSE, res_rel, IoU_NoS)
                val_loss = MAE
            else:
                log_dict = make_log('', locals(), F1, P, R)
                val_loss = F1
            if self.seg_metirx[0]:
                I, U, TPTN, POS, GT = torch.tensor(self.seg_metirx).sum(1).float()
                seg_IoU = I / (U + 0.00001)
                seg_acc = TPTN / ((len(self.seg_metirx[2]) * self.img_size ** 2) + 0.00001)
                seg_P = I / (POS + 0.00001)
                seg_R = I / (GT + 0.00001)
                seg_log = make_log('', locals(), seg_acc, seg_IoU, seg_P, seg_R)
                log_dict.update(seg_log)
        else:
            TP_p, TP_t, FP_p, FN_t = self.nos_collection_gt.cat_all()
            P_all, R_all, F1_all = PRF1(TP_p, TP_t, FP_p, FN_t)
            if self.params['training']['nos_task']:
                MAE_all, RMSE_all, res_rel_all, IoU_NoS_all = NoS_metric(TP_p, TP_t, FP_p, FN_t)
                log_dict_all = make_log('', locals(), MAE_all, F1_all, P_all, R_all, RMSE_all, res_rel_all, IoU_NoS_all)
                val_loss = MAE_all
            else:
                log_dict_all = make_log('', locals(), F1_all, P_all, R_all)
                val_loss = 0
            if self.seg_metirx_gt[0]:
                I, U, TPTN, POS, GT = torch.tensor(self.seg_metirx_gt).sum(1).float()
                seg_IoU_all = I / (U + 0.00001)
                seg_acc_all = TPTN / (len(self.seg_metirx_gt[2]) * self.img_size ** 2)
                seg_P_all = I / (POS + 0.00001)
                seg_R_all = I / (GT + 0.00001)
                seg_log_all = make_log('', locals(), seg_acc_all, seg_IoU_all, seg_P_all, seg_R_all)
                log_dict_all.update(seg_log_all)

            log_dict.update(log_dict_all)
        return {
            'val_loss': val_loss,
            'log': log_dict
        }

    def validation_epoch_end(self, output):
        out = self.get_val_result(output)
        self.reset_metric_collection()
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


    def test_epoch_end(self, output):
        re_dict = self.get_val_result(output)
        self.result = re_dict['log']
        if not self.gt_rpn_training:
            nos_p, nos_t, _, _ = self.nos_collection.cat_all(True)
            iou = self.iou_collection.cat_all(True)[0]
            score = self.score_collection.cat_all(True)[0]
            area = self.area_collection.cat_all(True)[1]

            tp_dict = dict(
                nos_gt=nos_t,
                nos_pre=nos_p,
                iou=iou,
                score=score,
                area=area
            )
            pd.DataFrame(tp_dict).to_csv(self.exp_folder + '/TP_data.csv')
        else:
            nos_p, nos_t, _, _ = self.nos_collection_gt.cat_all(True)
            score = self.score_collection_gt.cat_all(True)[0]
            pd.DataFrame(dict(
                nos_gt=nos_t,
                nos_pre=nos_p,
                score=score
            )
            ).to_csv(self.exp_folder + '/all_data.csv')
            self.reset_metric_collection()
        return re_dict

    def sample_step_end(self, variables):
        return


def get_match_inx(match_quality_matrix, iou_threshold=0.5):
    # match_quality_matrix.shape:(len(t_boxes), len(p_boxes))
    num_t, num_p = match_quality_matrix.shape
    gt2pre_inx = torch.full((num_t,), -1, dtype=torch.long)
    pre2gt_inx = torch.full((num_p,), -1, dtype=torch.long)
    iou_pre = torch.full((num_p,), -1.0)
    col_numel = match_quality_matrix.shape[1]
    while True:
        # 找出最IoU值最大的一对pre与gt, 获得最大值及其所在的行列号
        max_inx = match_quality_matrix.argmax()
        row = max_inx // col_numel
        col = max_inx % col_numel

        max_value = match_quality_matrix[row, col].item()
        # 若当前最大的IoU小于给定阈值，则结束匹配
        if max_value < iou_threshold:
            break
        # 为匹配到的gt/pre记录其对应的pre/gt的索引
        gt2pre_inx[row] = col
        pre2gt_inx[col] = row
        # 设置标记，该组配对将不再被匹配到
        match_quality_matrix[row] = -1
        match_quality_matrix[:, col] = -1
        # 记录匹配到的prediction的iou，未匹配到的值为-1
        iou_pre[col] = max_value
    # 转化为inx索引
    TP_pre_inx = gt2pre_inx[gt2pre_inx != -1]
    TP_gt_inx = torch.tensor([i for i, v in enumerate(gt2pre_inx) if v != -1]).long()
    FP_pre_inx = torch.where(pre2gt_inx == -1)[0]
    FN_gt_inx = torch.where(gt2pre_inx == -1)[0]
    return TP_pre_inx, TP_gt_inx, FP_pre_inx, FN_gt_inx, iou_pre


class result_collection(object):
    def __init__(self, key):
        self.key = key
        self.reset()

    def collect(self, target, pre, TP_pre_inx, TP_gt_inx, FP_p_inx, FN_t_inx):
        TP_p, TP_t, FP_p, FN_t = [torch.tensor([]) for i in range(4)]
        if target is not None:
            t_labels = target[self.key]
            TP_t, FN_t = t_labels[TP_gt_inx], t_labels[FN_t_inx]
        self.TP_t_label_all.append(TP_t)
        self.FN_t_label_all.append(FN_t)
        if pre is not None:
            p_labels = pre[self.key]
            TP_p, FP_p = p_labels[TP_pre_inx], p_labels[FP_p_inx]
        self.TP_p_label_all.append(TP_p)
        self.FP_p_label_all.append(FP_p)
        return TP_p, TP_t, FP_p, FN_t

    def reset(self):
        self.TP_t_label_all = []
        self.TP_p_label_all = []
        self.FP_p_label_all = []
        self.FN_t_label_all = []

    def cat_all(self, return_numpy=False):
        ret = list(map(torch.cat,
                       [to_cpu(self.TP_p_label_all), to_cpu(self.TP_t_label_all), to_cpu(self.FP_p_label_all),
                        to_cpu(self.FN_t_label_all)]))
        # ret = list(map(torch.cat, [to_cpu(self.TP_p_label_all), to_cpu(self.TP_t_label_all), to_cpu(self.FP_p_label_all), to_cpu(self.FN_t_label_all)]))

        return to_numpy(ret) if return_numpy else ret


def NoS_metric(TP_p, TP_t, FP_p=None, FN_t=None):
    '''return MAE, RMSE, res_rel, IoU_NoS'''
    has_nos = TP_t != 0
    # print('===============>', len(has_nos),'\n', TP_t,TP_p)
    valid_gt, valid_pre = TP_t[has_nos], TP_p[has_nos]
    if len(valid_gt) == 0:
        MAE, RMSE, res_rel, IoU_NoS = torch.tensor([9999] * 4)
        return MAE, RMSE, res_rel, IoU_NoS
    if isinstance(valid_gt, np.ndarray):
        valid_gt, valid_pre = torch.from_numpy(valid_gt), torch.from_numpy(valid_pre)
    valid_gt = valid_gt.cuda()
    # print(valid_pre.device, valid_gt.device)
    valid_pre = valid_pre.cuda()
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


def PRF1(TP_p, TP_t, FP_p, FN_t):
    assert len(TP_p) == len(TP_t)
    P = len(TP_p) / (len(TP_p) + len(FP_p) + 0.0000001)
    R = len(TP_p) / (len(TP_p) + len(FN_t) + 0.0000001)
    F1 = (2 * P * R) / (P + R) if P + R != 0 else 0
    P, R, F1 = torch.tensor([P, R, F1])
    return P, R, F1
