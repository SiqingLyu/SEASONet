import copy
import torchvision.models as models
from models.ISVnet import *
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
    if name == "ssnvvhnetu":  # 2022.5.25
        model = model(n_classes=n_classes, **param_dict)
    elif name == "ssnvvhnetu_classify":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "ssnvvhnetu_withfootprint":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "ssnvvhnetu_classify_withfootprint":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "ALL_VVHNetU_withfootprint":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "ALL_VVHNetU":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "vvhnetu":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "vvhnetu_withfootprint":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "maskrcnn_res50_ssnvvh":
        model = model(param_dict)
    elif name == "vvh":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "mux":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "ssn":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "ssnsar":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "muxssnvvh":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "muxssn":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "all_vvh_dsm":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "all_vvh_demdsm":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "all_vvh_centercat_dsmdem":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "all_vvh_dsmdem_catroad":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "all_vvh_dsmdem_catroad2":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "all_vvh_dsmdem_catroad3":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "all_vvh_dsmdem_FPcattoend":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "M3Net_SSN":
        model = model(n_classes=n_classes, **param_dict)
    return model


def _get_model_instance(name):
    try:
        return {
            "ssnvvhnetu": SSNVVHNetU,
            "ssnvvhnetu_classify": SSNVVHNetU_classify,
            "ssnvvhnetu_withfootprint": SSNVVHNetU_withfootprint,
            "ssnvvhnetu_classify_withfootprint": SSNVVHNetU_classify_withfootprint,
            "ALL_VVHNetU_withfootprint": ALL_VVHNetU_withfootprint,
            "ALL_VVHNetU": ALL_VVHNetU,
            "vvhnetu" : VVHNetU,
            "vvhnetu_withfootprint": VVHNetU_withfootprint,
            "maskrcnn_res50_ssnvvh": MaskRcnn_res50,
            "vvh": SAR_NetU,
            "mux": MUX_NetU,
            "ssn": SSN_NetU,
            "ssnsar": SSNSAR_NetU,
            "muxssnvvh": MUXSSNSAR_NetU,
            "muxssn": MUXSSN_NetU,
            "all_vvh_dsm": ALL_VVH_DSMNetU,
            "all_vvh_demdsm": ALL_VVH_DSMDEMNetU,
            "all_vvh_centercat_dsmdem": ALL_VVH_centercat_DSMDEMNetU,
            "all_vvh_dsmdem_catroad": ALL_VVH_DSMDEM_catROAD_NetU,
            "all_vvh_dsmdem_catroad2": ALL_VVH_DSMDEM_catROAD2_NetU,
            "all_vvh_dsmdem_catroad3": ALL_VVH_DSMDEM_catROAD3_NetU,
            "all_vvh_dsmdem_FPcattoend": ALL_VVH_DSMDEM_FPcattoendNetU,
            "M3Net_SSN": SSN_NetU_object

        }[name]
    except:
        raise ("Model {} not available".format(name))
