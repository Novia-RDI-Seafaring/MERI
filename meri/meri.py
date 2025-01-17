from docling.document_converter import DocumentConverter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import *
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from .intermediate_format.format_handler import HTMLFormatHandler

from .utils.docling_utils import export_to_html, vis_layout
from .extraction.extractor import JsonExtractor

class MERI:

    def __init__(self, pdf_path, extractor_kwargs={}, do_ocr = False, table_transformer_mode: TableFormerMode = TableFormerMode.ACCURATE, do_cell_matching: bool = True):
    
        self.pdf_path = pdf_path
        self.extractor_kwargs = extractor_kwargs
        self.format_handler = None

        table_structure_options = TableStructureOptions(mode=table_transformer_mode,
                                                        do_cell_matching=do_cell_matching)

        pipeline_options = PdfPipelineOptions(generate_picture_images=True,
                                                generate_table_images=True,
                                                do_ocr=do_ocr,
                                                table_structure_options=table_structure_options)
        backend = PyPdfiumDocumentBackend
        self.converter = DocumentConverter(
            format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options = pipeline_options, backend = backend)
                    }
            )   

    def to_intermediate(self):
        self.docling_result = self.converter.convert(self.pdf_path)

        self.int_format = export_to_html(self.docling_result.document)

        self.format_handler = HTMLFormatHandler(self.int_format)

    def vis_layout(self, **kwargs):

        return vis_layout(self.docling_result, **kwargs)
    
    def run(self, json_schema_str: str):
        if not self.format_handler:
            self.to_intermediate()
        
        self.jsonExtractor = JsonExtractor(intermediate_format=self.format_handler, **self.extractor_kwargs)
                
        return self.jsonExtractor.populate_schema(json_schema_string=json_schema_str)
