from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
from .transform import GeneralizedRCNNTransform
from .backbone_utils import resnet_fpn_backbone
from .generalized_rcnn import GeneralizedRCNN


class rpn_gt(nn.Module):
    def __init__(self):
        super(rpn_gt, self).__init__()
    def forward(self, images, features, targets=None):
        '''训练时返回空列表（roi_head在训练时追加真值box作为proposal，因此这里不需要再返回真值）,
        推理时返回box真值'''
        losses = {}
        if self.training:
            boxes = [torch.tensor([]).reshape(0,4).to(features['0']) for i in range(len(images.image_sizes))]
        else:
            boxes = [sample['boxes'] for sample in targets]
        return boxes, losses







class nosRCNN(GeneralizedRCNN):
    def __init__(self, backbone_name='resnet50', rpn=None, roi_head=None):
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        min_size=800
        max_size=1333
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        backbone = resnet_fpn_backbone(backbone_name, True)

        if rpn is None:
            rpn = rpn_gt()

        # if roi_head is None:
        #     roi_head =


