import json

from dataloaders.dataloaders import *

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "ssnvvh": UnetDataloader,
        "ssnvvh_bk255": UnetDataloader,
        "ssnvvh_classify": UnetDataloader,
        "ssnvvh_classify_withfootprint": UnetDataloader,
        "ssnvvh_withfootprint": UnetDataloader,
        "maskrcnn_ssnvvh": MaskRcnnDataloader

    }[name]
