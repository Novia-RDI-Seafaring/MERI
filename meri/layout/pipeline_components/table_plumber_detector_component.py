import deepdoctection as dd
from typing import List
from .utils import scale_coords, ProcessingService

class TablePlumberComponent(dd.PipelineComponent):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        
    def serve(self, dp: dd.Image) -> None:
        assert dp.pdfplumber_page

        source_height, source_width = dp.pdfplumber_page.height, dp.pdfplumber_page.width
        target_height, target_width, _ = dp._image.shape

        # Function to handle detection results
        def handle_detection_result(bbox):
            scaled_bbox = scale_coords(bbox, source_height, source_width, target_height, target_width)
            detect_results = dd.DetectionResult(
                box=scaled_bbox,
                class_name=dd.LayoutType.table, # Table class
                class_id=0
            )
            self.dp_manager.set_image_annotation(detect_results)

        # First use the built-in find_tables method
        tables = dp.pdfplumber_page.find_tables(
            table_settings={"vertical_strategy": "lines"}
            )
        for table in tables:
            handle_detection_result(table.bbox)

        # if not tables:
        # Then use the custom potential tables detection method regardless of previous results
        potential_tables = self.extract_potential_tables(dp.pdfplumber_page)
        print(f"Potential Tables Found: {len(potential_tables)}")
        for table in potential_tables:
            x0 = min(block['x0'] for block in table)
            top = min(block['top'] for block in table)
            x1 = max(block['x1'] for block in table)
            bottom = max(block['bottom'] for block in table)
            handle_detection_result((x0, top, x1, bottom))

    
    def extract_potential_tables(self, page):
        '''
        Extract potential tables from a page using pdfplumber
        Args:
            page: pdfplumber page object
        Returns:
            list of potential tables
        '''
        # Extract text blocks from the page
        text_blocks = page.extract_words(keep_blank_chars=True)
        # Sort the text blocks by top(vertical) and x0(horizontal) left coordinates
        text_blocks_sorted = sorted(text_blocks, key=lambda b: (b['top'], b['x0']))
        potential_tables = []
        current_table = [text_blocks_sorted[0]]
        
        for block in text_blocks_sorted[1:]:
            last_block = current_table[-1]
            # If the current block is close to the last one vertically, consider it part of the same table
            if (block['top'] - last_block['bottom']) < 15:
                current_table.append(block)
            else:
                # If the current block is far from the last one vertically, consider it a new table
                if len(current_table) > 3:
                    potential_tables.append(current_table)
                current_table = [block]
        # Check the last accumulated table
        if len(current_table) > 3:
            potential_tables.append(current_table)
        
        return potential_tables
 

    def get_meta_annotation(self):
        return dict([
            ("image_annotations", self.possible_categories()),
            ("sub_categories", {}),
            ("relationships", {}),
            ("summaries", []),
        ])
    
    def clone(self) -> 'TablePlumberComponent':
        return self.__class__(self.name)
    
    def possible_categories(self) -> List[dd.ObjectTypes]:
        return [dd.LayoutType.table]