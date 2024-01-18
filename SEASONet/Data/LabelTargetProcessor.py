"""
Date: 2022.9.22
Author: Lv
this class aims to create a class which contains all the operation on Labels
Last coded date: 2022.9.23
"""
import bbox_visualizer as bbv
from skimage.measure import label, regionprops
import numpy as np
from numpy import ndarray
import torch
import os
from typing import Optional
import cv2
import sys
sys.path.append('/home/dell/lsq/SEASONet_231030')
# sys.path.append('D:\python文件库\V100\LSQNetModel_on')
from pytorch_tools import *
from tools import make_dir
import tiffile as tif
# from Data.ImageProcessor import Labels


class LabelTarget:
    def __init__(self,
                 label_data: ndarray = None, height_data = None,
                 target_data: dict = None):
        assert label_data is not None or target_data is not None, 'Must input some Data!'
        self.label = label_data
        self.target = target_data
        self.height = height_data

    def from_target(self, background: int = 0):
        """
        get label data from a target
        :param background: default as 0
        :return: mask_all: label as numpy.ndarray
        """
        assert self.target is not None, 'Must input target data to get label'
        target_data = detach_all(self.target)
        data = target_data.copy()
        labels = data['nos']
        masks = data['masks']
        mask_all = np.zeros_like(masks[0])
        for i in range(len(masks)):
            mask_data = masks[i]
            label_data = labels[i]
            mask_all = np.where(mask_data != background, int(label_data), mask_all)
        return mask_all

    def to_target(self,
                  image_id: list = None,
                  file_name: str = None,
                  if_buffer_proposal = False,
                  buffer_pixels = 0,
                  buffer_assist = False,
                  img_size=128,
                  background: int = 0,
                  tan_factor: int = None,
                  **kwargs):
        assert self.label is not None, 'Must input label data to get target'
        boxes, masks, labels, areas, noses, heights = self.get_box_mask_value_area(**kwargs)
        assert len(boxes) > 0
        if if_buffer_proposal:
            for ii in range(len(boxes)):
                if buffer_assist == 'nos_assist':
                    buffer_pixels = int(np.ceil(noses[ii]/3 + 1))
                elif buffer_assist == 'pre_assist':
                    height = heights[ii]
                    buffer_pixels = int(np.around(height) / 3)  # for nos to buffer pixels 3(*3)~=10m/pixel
                elif buffer_assist == 'sup_assist':
                    height = heights[ii]
                    buffer_pixels = int(np.ceil(height / 9))  # for height to buffer pixels 9~=10m/pixel
                if tan_factor is not None:
                    buffer_pixels = int(np.around(buffer_pixels / (tan_factor)))
                # if buffer_assist:
                #     nos = np.mean(pre_data[masks[ii] != background])
                box = boxes[ii]
                x_min, y_min, x_max, y_max = box  # west, north, east, south
                xbuf_min, ybuf_min, xbuf_max, ybuf_max = x_min - 0, y_min - buffer_pixels, x_max + 0, y_max
                # xbuf_min, ybuf_min, xbuf_max, ybuf_max = x_min - 0, y_min, x_max + 0, y_max + buffer_pixels
                if ybuf_min >= ybuf_max:
                    ybuf_min = ybuf_max - 1
                y_min, x_min, y_max, x_max = ybuf_min if ybuf_min > 0 else 0,\
                                             xbuf_min if xbuf_min > 0 else 0, \
                                             ybuf_max if ybuf_max < img_size else img_size,\
                                             xbuf_max if xbuf_max < img_size else img_size  # row is y, column is x
                buffer_box_rc = [x_min, y_min, x_max, y_max]
                boxes[ii] = buffer_box_rc
        target = {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': torch.tensor(masks, dtype=torch.uint8),
            'area': torch.FloatTensor(areas),
            'iscrowd': torch.tensor([0] * len(boxes)),
            'image_id': torch.tensor(image_id),
            'nos': torch.FloatTensor(noses),
            'file_name': file_name
        }
        self.target = target
        return target

    def to_target_cpu(self, **kwargs):
        assert self.label is not None, 'Must input label data to get target'
        boxes, masks, labels, areas, noses, heights = self.get_box_mask_value_area(**kwargs)
        if boxes is None:
            return None
        assert len(boxes) > 0
        target = {
            'boxes': np.array(boxes),
            'labels': np.array(labels, dtype=np.int64),
            'masks': np.array(masks, dtype=np.uint8),
            'area': np.array(areas, dtype=np.float),
            'nos': np.array(noses,  dtype=np.float)
        }
        self.target = target
        return target

    def target_to_device(self, device: str = 'cpu'):
        assert device in ['cpu', 'cuda']
        assert self.target is not None, 'No target data found!'
        dict_to_device(self.target, device=device)

    def target_detach(self):
        contain = self.target
        if isinstance(contain, dict):
            ret = {}
            for k, v in contain.items():
                if hasattr(v, 'grad_fn') and v.grad_fn:
                    ret[k] = v.detach()
                else:
                    ret[k] = v
        self.target = ret
        return ret

    def get_box_mask_value_area(self,
                                area_thd: int = 1,
                                mask_mode: str = 'value',
                                background: int = 0,
                                label_is_value: bool = False,
                                value_mode: str = 'mean'):
        """
        use skimage.measure to get boxes, masks and the values, the areas of them
        :param label_is_value:
        :param background: background value of the image
        :param area_thd: objects whose area is below area_thd will be discard
        :param mask_mode: whether to connect pixels by 'is not background' or values
        :return: Boxes, Masks, Labels, Areas, all in array type
        """
        assert mask_mode in ['value', '01'], 'mask_mode must in [value, 01]'
        data = np.copy(self.label)
        height_data = np.copy(self.height) if self.height is not None else None
        value_region = label(data, connectivity=2, background=background)
        boxes, masks, labels, areas, nos_list, heights = [], [], [], [], [], []
        for region in regionprops(value_region):
            if region.area < area_thd: continue
            # region.bbox垂直方向为x， 而目标检测中水平方向为x
            y_min, x_min, y_max, x_max = region.bbox
            boxes.append([x_min, y_min, x_max, y_max])
            m = value_region == region.label
            if value_mode == 'argmax':
                # 取众数
                value = np.bincount(data[m]).argmax()
            if value_mode == 'mean':
                value = np.mean(data[m])

            if value_mode == 'meanheight':
                #from floor to height
                value = 3 * np.mean(data[m])

            if height_data is not None:
                height = np.mean(height_data[m])
                heights.append(height)

            nos_list.append(value)
            masks.append(m)
            labels.append(value if label_is_value else 1)
            areas.append(region.area)
        if len(boxes) == 0:
            return None, None, None, None, None, None
        assert background not in labels
        masks = np.array(masks)
        if mask_mode is '01':
            masks = np.where(masks, 1, 0)
        return np.array(boxes), masks, np.array(labels), np.array(areas), np.array(nos_list), np.array(heights)

    def get_box_mask_value_area_pre(self,
                                area_thd: int = 4,
                                mask_mode: str = 'value',
                                background: int = 0,
                                label_is_value: bool = False,
                                value_mode: str = 'argmax'):
        """
        use skimage.measure to get boxes, masks and the values, the areas of them
        :param label_is_value:
        :param background: background value of the image
        :param area_thd: objects whose area is below area_thd will be discard
        :param mask_mode: whether to connect pixels by 'is not background' or values
        :return: Boxes, Masks, Labels, Areas, all in array type
        """
        assert mask_mode in ['value', '01'], 'mask_mode must in [value, 01]'
        data = np.copy(self.pre)
        value_region = label(data, connectivity=2, background=background)
        boxes, masks, labels, areas, nos_list = [], [], [], [], []
        for region in regionprops(value_region):
            if region.area < area_thd: continue
            # region.bbox垂直方向为x， 而目标检测中水平方向为x
            y_min, x_min, y_max, x_max = region.bbox
            boxes.append([x_min, y_min, x_max, y_max])
            m = value_region == region.label
            if value_mode == 'argmax':
                # 取众数
                value = np.bincount(data[m]).argmax()
            if value_mode == 'mean':
                value = np.mean(data[m])
            nos_list.append(value)
            masks.append(m)
            labels.append(value if label_is_value else 1)
            areas.append(region.area)
        if len(boxes) == 0:
            return None, None, None, None, None
        assert background not in labels
        masks = np.array(masks)
        if mask_mode is '01':
            masks = np.where(masks, 1, 0)
        return np.array(boxes), masks, np.array(labels), np.array(areas), np.array(nos_list)

    def draw_target_on_image(self,
                             predicts: dict = None,
                             img_path: str = ''):
        assert os.path.isfile(img_path) is True, 'img_path must be a file path!'
        assert self.target is not None, 'Must initial with a target!'

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:, :, -3:]  # take RGB
        target_data = detach_all(self.target)
        boxes = target_data['boxes'].cpu()
        values_gt = target_data['nos'].cpu().numpy()
        box_color = (255, 0, 255)
        plot_labels = []
        pred_nos = []
        new_box = []  # use new box to transfer float type into int

        if predicts is not None:
            predicts = detach_all(predicts)
            scores = predicts['scores'].cpu().numpy()
            pred_nos = predicts['nos'].cpu().numpy()
            boxes = predicts['boxes'].cpu().numpy()
        for i in range(len(boxes)):
            box = []
            for j in range(4):
                box.append(int(np.round(boxes[i][j])))
            new_box.append(box)

            gt = str(values_gt[i])
            plot_label = ' ' + gt
            if predicts is not None:
                pred = str(np.round(pred_nos[i], 1))
                plot_label = ' ' + gt + ' | ' + pred + ' '
            plot_labels.append(plot_label)
        boxes = new_box
        img = bbv.draw_multiple_rectangles(img,
                                           boxes,
                                           bbox_color=box_color,
                                           thickness=1)
        img = bbv.add_multiple_labels(img,
                                      plot_labels,
                                      boxes,
                                      text_bg_color=box_color,
                                      text_color=(0, 0, 0))
        return img

    def draw_target_on_image_v2(self,
                                predicts: dict = None, if_round=False):
        """
        use v2 when the resolution of the image is too low to draw bbox on,
        this function is for assigning the nos value to the target masks,
        so that you can see the result in vision
        :param predicts:
        :param img_path:
        :return:
        """
        assert self.target is not None, 'Must initial with a target!'
        target_data = detach_all(self.target)
        masks = target_data['masks'].cpu().numpy()
        preds = predicts['nos'].cpu().numpy()
        pred_arr = np.zeros_like(masks[0])
        for i in range(len(masks)):
            pred = preds[i]
            mask = masks[i]
            if if_round:
                pred_arr = np.where(mask > 0, int(np.round(pred)), pred_arr)
            else:
                pred_arr = np.where(mask > 0, pred, pred_arr)
        return pred_arr

    def save_targetdraw_image(self,
                              save_path: str,
                              file_name: str = None,
                              opt: str = None,
                              if_round = False,
                              **kwargs):
        assert file_name is not None, 'File name must be input!'
        assert opt in ['v1', 'v2'], 'only two options(v1, v2) are available!'
        make_dir(save_path)
        save_path = save_path + '/' + file_name + '.tif'
        if opt is 'v1':
            img = self.draw_target_on_image(**kwargs)
            cv2.imwrite(save_path, img)
        if opt is 'v2':
            img = self.draw_target_on_image_v2(predicts=kwargs['predicts'], if_round=if_round)
            tif.imsave(save_path, img)

    def rewrite_label_with_areathd(self, save_path: str = None, filename: str = None, **kwargs):
        assert os.path.isdir(make_dir(save_path)), 'save path must be a dir!'
        if self.target is not None:
            lab = self.from_target()
        else:
            self.to_target_cpu(**kwargs)
            lab = self.from_target()
        tif.imsave(os.path.join(save_path, filename + '.tif'), lab)
