from .utils import parse_table_annotations

class TableDataset:
    """ Python iterator that iterates through the dataset. Each sample of the dataset contains:
    fitz_page, pdfplumber page, bounding box of table (as xyxy), gt table content, reference to .csv with gt table content.
    """

    def __init__(self, annotations_path, pdf_path, ann_json_name='annotation.json'):

        self.fitz_pages, self.plumber_pages, self.bboxes, self.table_contents, self.references = parse_table_annotations(
            annotations_path, 
            pdf_path,
            ann_json_name)

        assert len(self.fitz_pages) == len(self.bboxes) == len(self.table_contents) == len(self.plumber_pages)

        self.n = len(self.fitz_pages)
        self.current=0

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current <= self.n:
            return (self.fitz_pages[self.current-1],
                    self.plumber_pages[self.current-1],
                    self.bboxes[self.current-1],
                    self.table_contents[self.current-1],
                    self.references[self.current-1])

        raise StopIteration