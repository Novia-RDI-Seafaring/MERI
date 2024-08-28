import deepdoctection as dd
import os
from typing import List
from .utils import scale_coords

class WordUnionComponent(dd.PipelineComponent):
    ''' For given element x of type dd.LayoutType makes sure that all words are included in the bounding box.
    Finds word (dd.LayoutType result from e.g. OCR) bounding boxes that overlap with given element x bounding box.
    Merges bounding boxes of overlapping words and element x and updates element x's bounding box with this
    merged bounding box
    '''

    def __init__(self, unite: List[dd.LayoutType] = [dd.LayoutType.table]):
        self.unite = unite
        super().__init__(name=self.__class__.__name__)

    def serve(self, dp: dd.Image) -> None:
        
        # iterate over active annotations of elements that should be united with overlapping words
        for ann in filter(lambda x: x.category_name in self.unite and x.active, dp.annotations):
            bboxes_to_join = [ann.bounding_box]
            for word_ann in filter(lambda x: x.category_name == dd.LayoutType.word and x.active, dp.annotations):
                iou = dd.coco_iou(word_ann.bounding_box.to_np_array(mode='xyxy')[None], ann.bounding_box.to_np_array(mode='xyxy')[None])
                if iou>0:
                    bboxes_to_join.append(word_ann.bounding_box)
            merged_box = dd.merge_boxes(*bboxes_to_join)
            ann.bounding_box = merged_box

    def get_meta_annotation(self):
        return dict([
                ("image_annotations", self.possible_categories()),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", []),
            ])
    
    def clone(self) -> 'WordUnionComponent':
        return self.__class__(self.name)
    
    def possible_categories(self) -> List[dd.ObjectTypes]:
        return self.unite # can also register new objectTypes in registry!
    