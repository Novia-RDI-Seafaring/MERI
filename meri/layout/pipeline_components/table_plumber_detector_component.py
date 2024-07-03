import deepdoctection as dd
from typing import List
from .utils import scale_coords, ProcessingService

# https://stackoverflow.com/questions/76584489/pdfplumber-python-extract-tables-setting-for-the-specific-strategy
# https://github.com/jsvine/pdfplumber
def is_potential_table(text_blocks):
        '''
        Determines if a list of text blocks resembles a table structure.
        Args:
            text_blocks: list of text blocks
        Returns:
            bool: True if the text blocks resemble a table, False otherwise
        '''
        if len(text_blocks) < 4:
            return False
        
        # Check alignment and spacing patterns
        alignment_tolerance = 5 # text blocks that are within 5 units of each other horizontally are considered to be in the same column.
        column_positions = {} # Dictionary to store the number of text blocks aligned at each position.
        
        # Count the number of text blocks aligned at each position, and Group text blocks based on their horizontal positions (x0).
        for block in text_blocks:
            col_position = block['x0']
            column_found = False
            # Check if the block is close to any existing column position
            for pos in column_positions:
                # checks if the horizontal positions (x0 values) are identical or close to each other
                if abs(pos - col_position) < alignment_tolerance:
                    column_positions[pos] += 1
                    column_found = True
                    break
            # If no close column position found, add a new column position
            if not column_found:
                column_positions[col_position] = 1

        # Consider it a table if there are multiple text blocks aligned vertically
        num_columns = len([count for count in column_positions.values() if count > 1])
        
        # Additional density check: tables usually have fewer words per block on average
        average_words_per_block = sum(len(block['text'].split()) for block in text_blocks) / len(text_blocks)
        
        return num_columns > 1 and average_words_per_block < 5

class TablePlumberComponent(dd.PipelineComponent):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        
    def serve(self, dp: dd.Image) -> None:
        assert dp.pdfplumber_page

        source_height, source_width = dp.pdfplumber_page.height, dp.pdfplumber_page.width
        target_height, target_width, _ = dp._image.shape
        
        detected_boxes = [] # List to store detected boxes

        # # Function to handle detection results
        # def handle_detection_result(bbox):
        #     scaled_bbox = scale_coords(bbox, source_height, source_width, target_height, target_width)
        #     detect_results = dd.DetectionResult(
        #         box=scaled_bbox,
        #         class_name=dd.LayoutType.table, # Table class
        #         class_id=0
        #     )
        #     self.dp_manager.set_image_annotation(detect_results)
        #     detected_boxes.append(scaled_bbox) # Append the scaled bbox to the list
        
        # Function to handle detection results
        def handle_detection_result(bbox):
            scaled_bbox = scale_coords(bbox, source_height, source_width, target_height, target_width)
            # Check for overlaps with existing detections before adding new ones
            if not self.is_overlapping(scaled_bbox, detected_boxes):
                detect_results = dd.DetectionResult(
                    box=scaled_bbox,
                    class_name=dd.LayoutType.table,
                    class_id=0,
                    score=1
                )
                self.dp_manager.set_image_annotation(detect_results)
                detected_boxes.append(scaled_bbox)
            else:
                print(f"Skipped overlapping table: {scaled_bbox}")


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

    # Function to check if new_box overlaps with existing_boxes
    def is_overlapping(self, new_box, existing_boxes, threshold=0.1):
        # Check if new_box overlaps with any existing_box more than threshold percentage
        for box in existing_boxes:
            # Calculate intersection over union (IoU) here, and compare with threshold
            if self.calculate_iou(new_box, box) > threshold:
                return True
        return False
    
    # Function to calculate Intersection over Union (IoU) and check for overlap
    def calculate_iou(self, box1, box2):
        # Calculate the Intersection over Union (IoU) of two bounding boxes.
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])

        inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou
    
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
                if len(current_table) > 3 and is_potential_table(current_table): # NEW - Check if the current table is a potential table
                    potential_tables.append(current_table)
                current_table = [block]
        # Check the last accumulated table
        if len(current_table) > 3 and is_potential_table(current_table): # NEW - Check if the current table is a potential table
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