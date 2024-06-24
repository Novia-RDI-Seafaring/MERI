from ...utils import pil_to_base64, pdf_to_im
from .page_element import PageElement
from PIL import Image
import pdfplumber.page
import fitz
from typing import Tuple


class Figure(PageElement):
    """ Content class for Figure elements.
    """

    def __init__(self, pdf_bbox: Tuple[float,float,float,float], im: Image, fitz_page: fitz.Page, plumber_page: pdfplumber.page) -> None:
        """ Stores an image of the figure
        """
        super().__init__(pdf_bbox, fitz_page.number)
        self.detectiion_im = im
        self.fitz_page = fitz_page
        self.plumber_page = plumber_page

    def get_content(self):
        """ TODO return content of figure (e.g. image)
        """
        if self.content is not None:
            return self.content
        
        content = self.outer_image
        self.content = content
        return content #'return figure content'

    def as_markdown_str(self):
        """ Returns image of figure as markdown string. 
        """
        image = self.get_content()
        if image is None:
            return "![Figure image not found]()" 

        image_base64 = pil_to_base64(image)
        return f'![Figure](data:image/png;base64,{image_base64})'

    @property
    def outer_image(self):
        
        return pdf_to_im(self.fitz_page, self.outer_bbox)