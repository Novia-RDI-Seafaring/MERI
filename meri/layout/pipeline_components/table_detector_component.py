import deepdoctection as dd
import os
from typing import List
from .utils import scale_coords

class TableDetectorComponent(dd.PipelineComponent):
    ''' Dummy detecotor, could be used to add annotations from pymupdf for example
    '''

    def __init__(self):
        super().__init__(name=self.__class__.__name__)

    def serve(self, dp: dd.Image) -> None:
        
        # required fitz page property
        assert dp.pdfplumber_page

        tables = dp.pdfplumber_page.find_tables(table_settings={"vertical_strategy":"text"})
        
        source_height, source_width = dp.pdfplumber_page.height, dp.pdfplumber_page.width
        target_height, target_width, _ = dp._image.shape

        for table in tables:
            bbox = scale_coords(table.bbox, source_height, source_width, target_height, target_width) #list(rect)#

            detect_results = dd.DetectionResult(
                box = bbox,
                class_name = dd.LayoutType.table, #ExtendedLayoutType.drawing,
                class_id=0
            )

            self.dp_manager.set_image_annotation(detect_results)

    def get_meta_annotation(self):
        return dict([
                ("image_annotations", self.possible_categories()),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", []),
            ])
    
    def clone(self) -> 'TableDetectorComponent':
        return self.__class__(self.name)
    
    def possible_categories(self) -> List[dd.ObjectTypes]:
        return [dd.LayoutType.table] # can also register new objectTypes in registry!
    