import deepdoctection as dd
import os
from typing import List
import fitz


class DummyDetectorComponent(dd.PipelineComponent):
    ''' Dummy detecotor, could be used to add annotations from pymupdf for example
    '''

    def serve(self, dp: dd.Image) -> None:
        detect_results = dd.DetectionResult(
            box = [10,10,20,20],
            class_id = 0,
            score= 0,
            class_name = dd.LayoutType.table,
        )
        self.dp_manager.set_image_annotation(detect_results)

    def get_meta_annotation(self):
        return dict([
                ("image_annotations", self.possible_categories()),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", []),
            ])
    
    def clone(self) -> 'DummyDetectorComponent':
        return self.__class__(self.name)
    
    def possible_categories(self) -> List[dd.ObjectTypes]:
        return [dd.LayoutType.table] # can also register new objectTypes in registry!
    