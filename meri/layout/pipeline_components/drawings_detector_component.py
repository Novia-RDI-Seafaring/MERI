import deepdoctection as dd
import os
from typing import List
from .utils import scale_coords

class DrawingsDetectorComponent(dd.PipelineComponent):
    ''' Dummy detecotor, could be used to add annotations from pymupdf for example
    '''

    def __init__(self):
        super().__init__(self.__class__.__name__)

    def serve(self, dp: dd.Image) -> None:
        
        # required fitz page property
        assert dp.fitz_page

        d = dp.fitz_page.get_drawings()  # Extract drawings from the page
        new_rects = []
        for p in d:
            if p["rect"].is_empty:
                continue
            w = p["width"] if p["width"] is not None else 0 #1  # Use a default width of 1 if width is None
            r = p["rect"] + (-w, -w, w, w)  # Enlarge each rectangle by width value

            # Merging and filtering logic
            merged = False
            for i, existing_rect in enumerate(new_rects):
                if r.intersects(existing_rect):  # Check if rectangles intersect
                    new_rects[i] = existing_rect | r  # Merge rectangles
                    merged = True
                    break
            if not merged and not any(r in existing for existing in new_rects):
                new_rects.append(r)

        # Remove duplicates and fully contained rectangles
        final_rects = []
        for r in new_rects:
            if all(not (r in other and r != other) for other in new_rects):
                final_rects.append(r)
        
        source_height, source_width = dp.fitz_page.rect.height, dp.fitz_page.rect.width
        target_height, target_width, _ = dp._image.shape

        for rect in final_rects:
            
            if r.width <= 15 or r.height <= 15:
                continue  # Skip thin lines or very small drawings
            
            # TODO rect are based on pdf coordinates, transform to image coordinates
            rect_tr = scale_coords(rect, source_height, source_width, target_height, target_width) #list(rect)#
            detect_results = dd.DetectionResult(
                box = rect_tr,
                class_name = dd.LayoutType.figure, #ExtendedLayoutType.drawing,
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
    
    def clone(self) -> 'DrawingsDetectorComponent':
        return self.__class__(self.name)
    
    def possible_categories(self) -> List[dd.ObjectTypes]:
        return [dd.LayoutType.figure] # can also register new objectTypes in registry!
    