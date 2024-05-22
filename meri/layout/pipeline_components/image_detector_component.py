import deepdoctection as dd
import os
from typing import List
from .utils import scale_coords

class ImageDetectorComponent(dd.PipelineComponent):
    ''' Dummy detecotor, could be used to add annotations from pymupdf for example.
    Score of detected images from pdf is always 1.
    '''

    def __init__(self):
        super().__init__(name=self.__class__.__name__)

    def serve(self, dp: dd.Image) -> None:
        
        # required fitz page property
        assert dp.fitz_page

        images_info = dp.fitz_page.get_image_info()
        
        source_height, source_width = dp.fitz_page.rect.height, dp.fitz_page.rect.width
        target_height, target_width, _ = dp._image.shape

        for info in images_info:
            bbox = scale_coords(info['bbox'], source_height, source_width, target_height, target_width) #list(rect)#

            detect_results = dd.DetectionResult(
                box = bbox,
                class_name = dd.LayoutType.figure, #ExtendedLayoutType.drawing,
                class_id=0,
                score=1.0
            )

            self.dp_manager.set_image_annotation(detect_results)

    def get_meta_annotation(self):
        return dict([
                ("image_annotations", self.possible_categories()),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", []),
            ])
    
    def clone(self) -> 'ImageDetectorComponent':
        return self.__class__(self.name)
    
    def possible_categories(self) -> List[dd.ObjectTypes]:
        return [dd.LayoutType.figure] # can also register new objectTypes in registry!
    