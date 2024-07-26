from abc import ABC, abstractmethod
from typing import Tuple, List
from ...utils import merge_bboxes

class PageElement(ABC):
    """ Base class for all content elements
    """

    def __init__(self, pdf_bbox: Tuple[float,float,float,float], page_idx, seqno=None) -> None:
        from .textblock import TextBlock

        self.pdf_bbox = pdf_bbox
        self.seqno = seqno
        self.children: List[TextBlock] = []
        self.page_idx = page_idx
        self.content = None
    
    @abstractmethod
    def get_content(self):
        """ Returns content of content element. e.g. for text the text, for tables the table information in some format, e.g. dataframe
        """
        pass
    
    @abstractmethod
    def as_markdown_str(self):
        """ Convert elements content into markdown. returns markdown string
        """
        pass

    def get_order(self):
        """ Returns position in reading order. low = comes first , high = later
        """
        if self.seqno:
            return self.seqno
        else:
            if len(self.children) == 0:
                return None
            else:
                seqno = sum([child.seqno for child in self.children])/len(self.children)
                return seqno
        
    def add_children(self, elements: List['PageElement']):
        """ Add childelements.
        """
        self.children += elements
    
    @property
    def attribute_str(self):
        atts = {
            "bbox": self.pdf_bbox,
            "page_index": self.page_idx
        }
        atts_str = ' '.join(f'{key}={value}' for key, value in atts.items())

        return atts_str
    
    """ 
    @property
    def bbox_html_comment(self) -> str:
        
        return f"<!-- Bounding box (x0,y0,x1,y1): {self.pdf_bbox} Page index: {self.page_idx} -->"
    """

    @property
    def outer_bbox(self):
        """We assume that all textblocks that are overlapping with the detected bbox, should actually be contained in the bbox. 
        Therefore this function merges the detected bbox with all overlapping textblock boxes. Especially true for tables,
        because sometimes the table bbox is not highly accurate and might crop of some of the tables content.
        """
        boxes_to_merge = [self.pdf_bbox] + [child.pdf_bbox for child in self.children]
        return merge_bboxes(boxes_to_merge)