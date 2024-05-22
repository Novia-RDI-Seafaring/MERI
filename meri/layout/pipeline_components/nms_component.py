import deepdoctection as dd
import os
from .utils import CONFIGS_PATH

class NMSComponent(dd.AnnotationNmsService):
    ''' Applied NMS. Elements in specification need to have score property in annotations
    '''

    def __init__(self, cfg_path='nms.yaml'):

        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(CONFIGS_PATH, cfg_path)

        cfg = dd.set_config_by_yaml(cfg_path)

        if not isinstance(cfg.LAYOUT_NMS_PAIRS.COMBINATIONS, list) and not isinstance(
            cfg.LAYOUT_NMS_PAIRS.COMBINATIONS[0], list
        ):
            raise ValueError("LAYOUT_NMS_PAIRS mus be a list of lists")

        super().__init__(cfg.LAYOUT_NMS_PAIRS.COMBINATIONS, cfg.LAYOUT_NMS_PAIRS.THRESHOLDS, cfg.LAYOUT_NMS_PAIRS.PRIORITY)