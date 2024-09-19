import deepdoctection as dd
from typing import List
from .utils import ProcessingService
from ultralytics import YOLO
from ..settings import CustomLayoutTypes
from deepdoctection import ModelDownloadManager, ModelCatalog
import os
from huggingface_hub import hf_hub_download

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
        elif method == 'yolov10doclay':
            
            layout_detector = YoloDetector(cfg)
            kwargs = {}
        else:
            raise NotImplementedError

        super().__init__(layout_detector=layout_detector, **kwargs)

    def serve(self, dp: dd.Image) -> None:

        if self.cover_prev_anns:
            self.cover_annotations(dp)
            
        super().serve(dp)

        if self.cover_prev_anns:
            self.uncover_annotations(dp)


class YoloDetector(dd.ObjectDetector):

    def __init__(self, cfg):
        super().__init__()
        self.categories = {obj.name: obj.value for obj in CustomLayoutTypes}
        mode = "LAYOUT"
        weights = (
            getattr(cfg.TF, mode).WEIGHTS
            if cfg.LIB == "TF"
            else (getattr(cfg.PT, mode).WEIGHTS)
        )

        absolute_path_weights = ModelCatalog.get_full_path_weights(weights)
        profile = ModelCatalog.get_profile(weights)
        if not os.path.exists(absolute_path_weights):
            print("loading weights from hf")
            weights = hf_hub_download(repo_id=profile.hf_repo_id, filename=profile.hf_model_name,
                                    local_dir=os.path.abspath(os.path.join(absolute_path_weights, os.pardir)))

        self.name = "YoloDetector"
        self.model_id = self.get_model_id()

        self.model = YOLO(absolute_path_weights)
        self.device = cfg.DEVICE

        self.filter_categories = getattr(getattr(cfg.PT, "LAYOUT"), "FILTER")

        self.model.to(self.device)

        converted_class_names = {key: value.replace('-', '_').lower() for key, value in self.model.names.items()}
        self.dd_aligned_class_names = {key: ('figure' if value == 'picture' else value) for key, value in converted_class_names.items()}



    def predict(self, image):
        result = self.model.predict(image)[0]
        detections = []
        for box, scores, pred_cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):  # xyxy format7
            class_name = self.dd_aligned_class_names[pred_cls.item()]
            if class_name not in self.filter_categories:
                detections.append(dd.DetectionResult(
                    box=box.tolist(),
                    score=scores.item(),
                    class_id=pred_cls.item(),
                    class_name=class_name
                ))
               
        return detections
    
    def possible_categories(self):
        return list(CustomLayoutTypes)
        #return list(self.dd_aligned_class_names.values())

    @classmethod
    def get_requirements(cls) -> List[dd.utils.detection_types.Requirement]:
        return [dd.utils.file_utils.get_pytorch_requirement()]

    def clone(self) -> 'YoloDetector':
        return self.__class__()

    def get_model_id(self) -> str:
        return dd.utils.identifier.get_uuid_from_str(self.name)[:8]