import numpy as np
import deepdoctection as dd
from PIL import Image, ImageDraw
from typing import List, Tuple
import os


class ProcessingService:

    def __init__(self) -> None:
        self.tmp_image = None

    def cover_annotations(self, dd_image: dd.Image):
        ''' In place adjustment of dd_page
        annotations: image.get_annotations() from deepdoctection.
        '''
        self.tmp_image = dd_image.image

        anns = dd_image.get_annotation()

        original_np_im = dd_image.image # get original image as np

        pil_image = Image.fromarray(original_np_im)
        im_draw= ImageDraw.Draw(pil_image)

        for ann in anns:
                #get_bounding_box, ann.bbox
                im_draw.rectangle(ann.get_bounding_box().to_np_array(mode='xyxy'), outline='white', fill='white')
        
        dd_image.image = np.asarray(pil_image)
    
    def uncover_annotations(self, dd_image: dd.Image):

        if self.tmp_image is None:
             raise Exception('cover annotations needs to be called before uncover annotations')

        dd_image.image = self.tmp_image

def scale_coords(source_coords, source_height, source_width, target_height, target_width):
        '''Transforms source coordinates (x0, y0, x1, y1)
        to target coordinates (x0,y0, x1,y1)'''

        x0, y0, x1, y1 = source_coords

        x0_rel = x0/source_width
        x1_rel = x1/source_width

        y0_rel = y0/source_height
        y1_rel = y1/source_height

        #rect_shape = [int(x0_rel*target_width+0.5),int(y0_rel*target_height+0.5), int(x1_rel*target_width+0.5), int(y1_rel*target_height+0.5)]
        rect_shape = [int(x0_rel*target_width),int(y0_rel*target_height), int(x1_rel*target_width), int(y1_rel*target_height)]

        return rect_shape