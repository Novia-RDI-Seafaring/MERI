import deepdoctection as dd
import os
from typing import List
from .utils import scale_coords

class TextDetectorComponent(dd.PipelineComponent):
    ''' Dummy detecotor, could be used to add annotations from pymupdf for example
    '''

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        

    def serve(self, dp: dd.Image) -> None:
        
        # required fitz page property
        assert dp.fitz_page and dp.fitz_page_dict

        target_height, target_width, _ = dp._image.shape

        seqnos = []
        for item in dp.fitz_page.get_text_blocks():
            bbox = scale_coords(item[:4], dp.fitz_page_dict['height'], dp.fitz_page_dict['width'], target_height, target_width)
            detect_results = dd.DetectionResult(
                box = bbox,
                class_name = dd.LayoutType.text,
                class_id=0,
                score = 1,
                text = item[4],
                relationships={
                    'seqno': item[5]
                }           
                )
            
            self.dp_manager.set_image_annotation(detect_results)
        self.seqnos=seqnos

    def get_meta_annotation(self):
        return dict([
                ("image_annotations", self.possible_categories()),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", []),
            ])
    
    def clone(self) -> 'TextDetectorComponent':
        return self.__class__(self.name)
    
    def possible_categories(self) -> List[dd.ObjectTypes]:
        return [dd.LayoutType.text] # can also register new objectTypes in registry!
    