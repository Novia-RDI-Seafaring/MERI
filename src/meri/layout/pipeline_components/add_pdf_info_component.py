import deepdoctection as dd
import os
from typing import List
import fitz
import pdfplumber


class AddPDFInfoComponent(dd.PipelineComponent):
    ''' Adds fitz page and extracted dictionary and pdfplumber page from this page to datapoint that is passed through pipeline.
    All following pipeline components can access both properties via the datapoint (Image class).
    e.g.
    dps = pipeline.analyye(pdf_path)
    dps[0].image_orig.fitz_page
    '''

    def __init__(self):
        super().__init__(self.__class__.__name__)

    def serve(self, dp: dd.Image):

        dp.fitz_page = self._load_fitz_page(dp.location, dp.page_number)# load fitz page from pdf path and page number
        
        textpage = dp.fitz_page.get_textpage()
        dp.fitz_page_dict = textpage.extractDICT()

        #also load page as pdfplumber
        dp.pdfplumber_page = pdfplumber.open(dp.location).pages[dp.page_number]

    def _load_fitz_page(self, pdf_path, page_number):

        doc = fitz.open(pdf_path)

        return doc[page_number]
    
    def get_meta_annotation(self):
        return dict([
                ("image_annotations", []),
                ("sub_categories", {}),
                ("relationships", {}),
                ("summaries", []),
            ])

    def clone(self) -> 'AddPDFInfoComponent':
        return self.__class__(self.name)