import copy
import torchvision.models as models
from models.detection.mask_rcnn import *
from models.SEASONet import *

def get_model(model_dict, n_maxdisp=256, n_classes=1, version=None, Maskrcnn = False):
    name = model_dict["model"]["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    if name == "SEASONet":
        model = model(param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "SEASONet": MaskRcnn_res50,
        }[name]
    except:
        raise ("Model {} not available".format(name))
