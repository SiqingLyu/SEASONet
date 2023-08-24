import copy
import torchvision.models as models
from models.detection.mask_rcnn import *
from models.MaskRcnn_res50 import *

def get_model(model_dict, n_maxdisp=256, n_classes=1, version=None, Maskrcnn = False):
    if Maskrcnn:
        name = model_dict["model"]["arch"]
        model = _get_model_instance(name)
        param_dict = copy.deepcopy(model_dict)
        # param_dict.pop("arch")
    else:
        name = model_dict["arch"]
        model = _get_model_instance(name)
        param_dict = copy.deepcopy(model_dict)
        param_dict.pop("arch")
    if name == "maskrcnn_res50_ssnvvh":
        model = model(param_dict)
    return model


def _get_model_instance(name):
    try:
        return {
            "maskrcnn_res50_ssnvvh": MaskRcnn_res50,
        }[name]
    except:
        raise ("Model {} not available".format(name))
