import deepdoctection as dd
from typing import List
from .utils import ProcessingService


class LayoutDetectorComponent(dd.ImageLayoutService, ProcessingService):
    ''' Execute ML-based layout analysis. Provide method to specify applied detection approach.

    Supports:
    - d2layout: detects tables, text, figures, lists 
    - doctr_textdetector: detects words (required to apply doctr text recognition afterwards)
    - DETR: only table detection https://huggingface.co/microsoft/table-transformer-detection
    '''

    def __init__(self, cfg_path, cover_prev_anns=False, method='d2layout', config_overwrite=[]):

        self.cover_prev_anns = cover_prev_anns
        config_overwrite = [] if config_overwrite is None else config_overwrite

        cfg = dd.set_config_by_yaml(cfg_path)

        if config_overwrite:
            cfg.update_args(config_overwrite)
        
        print('Running configuration: ', cfg.to_dict)
        if method == 'd2layout' or method == 'detr':
            # detects layout components like tables and figures
            
            cfg.freeze(freezed=False)
            kwargs = {}
            layout_detector = dd.build_detector(cfg, "LAYOUT")

        elif method == 'doctr_textdetector':
            # detects words

            path_weights_tl = dd.ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.DOCTR_WORD.PT)
            categories = dd.ModelCatalog.get_profile(cfg.WEIGHTS.DOCTR_WORD.PT).categories
            layout_detector = dd.DoctrTextlineDetector("db_resnet50", path_weights_tl, categories, cfg.DEVICE)
            kwargs = {'to_image': True, 'crop_image': True}

        else:
            raise NotImplementedError

        super().__init__(layout_detector=layout_detector, **kwargs)

    def serve(self, dp: dd.Image) -> None:

        if self.cover_prev_anns:
            self.cover_annotations(dp)
            
        super().serve(dp)

        if self.cover_prev_anns:
            self.uncover_annotations(dp)