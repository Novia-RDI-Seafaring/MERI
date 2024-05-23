import tqdm
import fitz
import pdfplumber
import itertools
from .elements import TextBlock, PageElement, Figure, Table, get_font_sizes, add_font_info_to_textblocks
from ..utils import scale_coords
from typing import List
import logging
import deepdoctection as dd
from PIL import Image
import numpy as np
from enum import Enum

class Format(Enum):
    MARKDOWN = 'markdown'

class DocumentTransformer:

    def __init__(self, pdf_path: str, table_extraction_method='llm') -> None:
        
        self.fitz_document = fitz.open(pdf_path)
        self.plumber_document = pdfplumber.open(pdf_path)
        self.pages: List[PageTransformer] = [PageTransformer(fitz_page, 
                                                             plumber_page,
                                                             table_extraction_method=table_extraction_method) for (fitz_page, plumber_page) in list(zip(self.fitz_document, self.plumber_document.pages))]

    @property
    def unmatched_text_blocks(self):
        return list(itertools.chain(*[page.unmatched_text_blocks for page in self.pages]))

    def transform_to(self, format: Format):
        
        if format == Format.MARKDOWN:
            page_markdowns: List[str] = []
            for page_transformer in self.pages:
                page_markdowns.append(page_transformer.to_markdown())

            return ' '.join(page_markdowns)
        
        else:
            print(format)
            raise NotImplementedError

    def docorate_unmatched_textblocks(self):
        """add annoattions lioke heading, body etc to textBlocks.
        Compute distribution of font size over pages. and based on this do decoration
        """
        font_size_dict = get_font_sizes(self.unmatched_text_blocks, setattrs=True)
        add_font_info_to_textblocks(self.unmatched_text_blocks, font_size_dict)
    
    def merge_with_annotations(self, dps: dd.datapoint.view.Page, match_types=List[dd.LayoutType]) -> None:
        """ matches annotations for each page
        """
        
        assert len(self.pages) == len(dps)

        for (page, dp) in tqdm.tqdm(list(zip(self.pages, dps))):
            page.match_with_annotations(dp, match_types)

        #logger.info(f'Matched Annotations for all ({len(dps)}) pages')

class PageTransformer:
    """ Class that enables fusion of raw pdf information with deepdoctection annotations.
    Class matches raw pdf information (textblocks) with annotations. Advantage: it is okay if 
    deepdoctection pipeline does not get all content.
    """

    def __init__(self, fitz_page: fitz.Page, plumber_page: pdfplumber.page, table_extraction_method = 'llm') -> None:
        self.fitz_page = fitz_page
        self.plumber_page = plumber_page

        self.table_extraction_method = table_extraction_method

        # initialize it with all raw content. After each match, respective textblock is removed from
        # unmatched_text_blocks
        self.unmatched_text_blocks: List[TextBlock] = self.raw_content(fitz_page)

        # Either Table or Image with matching textblocks as children
        self.elements: List[PageElement] = []

    @classmethod
    def raw_content(cls, page: fitz.Page):
        """ 
        """
        text_blocks = page.get_text_blocks()
        content = []
        for (x0, y0, x1, y1, _, block_no, block_type) in text_blocks:
            bbox = [x0,y0,x1,y1]
            textpage = page.get_textpage(clip=bbox)
            content.append(TextBlock(bbox, block_no, textpage))

        return content

    def to_markdown(self):
        """ Converts page to markdown
        """
        
        markdown_strs = []
        for element in self.get_content():
            if isinstance(element, TextBlock):
                markdown_strs.append(element.as_markdown_str())
            elif isinstance(element, Figure):
                markdown_strs.append(element.as_markdown_str())
            elif isinstance(element, Table):
                markdown_strs.append(element.as_markdown_str())
       
        return ' '.join(markdown_strs)

    # def get_content(self):
    #     """ Combines matched elements and unmatched text blocks and sorts it by reading order
    #     """
    #     content = self.elements + self.unmatched_text_blocks #
    #     return sorted(content, key = lambda element: element.get_order())
    def get_content(self):
        """ Combines matched elements and unmatched text blocks and sorts it by reading order """
        content = self.elements + self.unmatched_text_blocks
        return sorted(content, key=lambda element: element.get_order() if element.get_order() is not None else float('inf'))

    def match_with_annotations(self, dp: dd.datapoint.view.Page, match_types=List[dd.LayoutType]) -> None:
        """ Takes annotations from deepdoctection pipeline and tries to match the elements (currently only table and figure)
        to the textblocks extracted from the raw pdf. If one or multiple textblocks can be matched to an deepdoctection
        annotation they are added as children to this element.

        dp: page result from deepdoctection pipeline. e.g.
            pipeline = Pipeline.from_config(cfg_path='/workspaces/ai-information-extraction/layout_analysis/src/config/good_pipeline.yaml')
            pipeline.build()
            dps, page_dicts = pipeline.run(pdf_path)
            dp = dps[0]
        match_types: List of layout types that should be matched with Textblocks
        """
        pil_im = Image.fromarray(dp.image_orig.image)

        matched_text_block_ids = []
        for match_type in match_types:
            

            source_height, source_width, _ = dp.image.shape
            pdf_height, pdf_width = self.fitz_page.rect.height, self.fitz_page.rect.width

            elements = []
            for rel_ann in filter(lambda x: x.category_name == match_type, dp.get_annotation()):
                bbox = scale_coords(rel_ann.bbox, source_height, source_width, pdf_height, pdf_width)
                cropped_im = pil_im.crop(rel_ann.bbox)
                if match_type == dd.LayoutType.figure:
                    
                    element = Figure(bbox, im=cropped_im, fitz_page=self.fitz_page, plumber_page=self.plumber_page)
                elif match_type == dd.LayoutType.table:
                    element = Table(bbox, im=cropped_im, fitz_page=self.fitz_page, plumber_page=self.plumber_page,
                                    method=self.table_extraction_method)
                else:
                    raise NotImplementedError
            
                # match bboxes -> find all bboxes in page_content.text_blocks that are contained/overlap
                for i, text_block in enumerate(self.unmatched_text_blocks):
                    iou = dd.coco_iou(np.array(text_block.pdf_bbox)[None], np.array(bbox)[None])
                    if iou>0:
                        element.add_children([text_block])
                        matched_text_block_ids.append(i)
                elements.append(element)

            self.elements += elements

        self.unmatched_text_blocks = [self.unmatched_text_blocks[idx] for idx in list(range(len(self.unmatched_text_blocks))) if idx not in list(set(matched_text_block_ids))]
