import json
from typing import List, Tuple
import fitz
from layout_eval.utils import coco_to_xyxy, rescale_bbox, add_noise_to_bbox, add_noise_to_detected_bboxes


class LayoutEvaluator:
    def __init__(self, pdf_path: str, annotation_path: str):
        self.pdf_path = pdf_path
        self.annotation_path = annotation_path
        self.ground_truth_bboxes = self.load_cvat_annotations(annotation_path)
        self.detected_bboxes = []

    def load_cvat_annotations(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        image_dimensions = {image['id']: (image['width'], image['height']) for image in data['images']}
        ground_truth_bboxes = [(ann['image_id'], coco_to_xyxy(ann['bbox']), image_dimensions[ann['image_id']]) for ann in data['annotations'] if ann['category_id'] == 1]  # Filter for tables only
        return ground_truth_bboxes

    def extract_detected_bboxes(self, doc_transformer) -> List[Tuple[int, List[float]]]:
        """
        Extracts the detected bounding boxes from the document transformer.
        
        Parameters:
            doc_transformer: The document transformer containing the detected elements.
            
        Returns:
            List of tuple: The detected bounding boxes [(page_number, bbox), ...].
        """
        detected_bboxes = []
        for page_number in range(len(doc_transformer.pages)):
            page = doc_transformer.pages[page_number]
            for element in page.elements:
                if hasattr(element, 'pdf_bbox'):
                    detected_bboxes.append((page_number + 1, element.pdf_bbox))  # Page number starts from 1
        self.detected_bboxes = detected_bboxes
        return detected_bboxes
    
    def extract_noisy_detected_bboxes(self, doc_transformer, noise_level=0.05):
        """
        Extracts and adds noise to the detected bounding boxes.
        
        Parameters:
            doc_transformer: The document transformer containing the detected elements.
            noise_level (float): The noise level to add to the detected bounding boxes.
            
        Returns:
            List of tuple: The detected bounding boxes with added noise.
        """
        detected_bboxes = []
        for page_number in range(len(doc_transformer.pages)):
            page = doc_transformer.pages[page_number]
            for element in page.elements:
                if hasattr(element, 'pdf_bbox'):
                    detected_bboxes.append((page_number + 1, element.pdf_bbox))  # Page number starts from 1
        noisy_detected_bboxes = add_noise_to_detected_bboxes(detected_bboxes, noise_level)
        self.detected_bboxes = noisy_detected_bboxes
        return noisy_detected_bboxes

    @staticmethod
    def bbox_intersection_area(boxA, boxB):
        """
        Calculates the area of intersection between two bounding boxes.
        
        Parameters:
            boxA (list): The first bounding box in [x1, y1, x2, y2] format.
            boxB (list): The second bounding box in [x1, y1, x2, y2] format.
            
        Returns:
            float: The area of intersection between the two bounding boxes.   
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        if xA < xB and yA < yB:
            return (xB - xA) * (yB - yA)
        else:
            return 0

    @staticmethod
    def bbox_union_area(boxA, boxB, intersection_area):
        """
        Calculates the area of union between two bounding boxes.
        
        Parameters:
            boxA (list): The first bounding box in [x1, y1, x2, y2] format.
            boxB (list): The second bounding box in [x1, y1, x2, y2] format.
            intersection_area (float): The area of intersection between the two bounding boxes.
            
        Returns:
            float: The area of union between the two bounding boxes.
        """
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return areaA + areaB - intersection_area

    def compare_bounding_boxes(self):
        """
        Compares the detected bounding boxes with the ground truth bounding boxes.
        
        Returns:
            tuple: The overlapping bounding boxes and the comparison results.
        """
        rescaled_detected_bboxes = []
        doc = fitz.open(self.pdf_path)
        for bbox in self.detected_bboxes:
            page_number = bbox[0]
            page = doc.load_page(page_number - 1)
            new_scale = page.rect.width, page.rect.height
            # rescaled_bbox = rescale_bbox(bbox[1], old_scale=new_scale, new_scale=new_scale)
            rescaled_bbox = bbox[1]
            rescaled_detected_bboxes.append((page_number, rescaled_bbox))

        comparison_results = []

        # Use a set to keep track of matched ground truth boxes to avoid duplicate matches
        matched_gt_boxes = set()

        for pdf_box in self.detected_bboxes: # rescaled_detected_bboxes:
            pdf_page_num = pdf_box[0]
            pdf_bbox = pdf_box[1]
            best_iou = 0
            best_gt_box = None

            for cvat_box in self.ground_truth_bboxes:
                cvat_page_num = cvat_box[0]
                if cvat_page_num != pdf_page_num:
                    continue
                cvat_bbox = rescale_bbox(cvat_box[1], old_scale=cvat_box[2], new_scale=new_scale)
                intersection_area = self.bbox_intersection_area(pdf_bbox, cvat_bbox)
                union_area = self.bbox_union_area(pdf_bbox, cvat_bbox, intersection_area)
                if union_area > 0:
                    iou = intersection_area / union_area
                else:
                    iou = 0
                if iou > best_iou:
                    best_iou = iou
                    best_gt_box = cvat_box

            # if best_gt_box and best_gt_box not in matched_gt_boxes:
            if best_gt_box and (best_gt_box[0], tuple(best_gt_box[1]), tuple(best_gt_box[2])) not in matched_gt_boxes:
                matched_gt_boxes.add((best_gt_box[0], tuple(best_gt_box[1]), tuple(best_gt_box[2])))
                # matched_gt_boxes.add(best_gt_box)
                comparison_results.append({
                    'pdf_page_num': pdf_page_num,
                    'cvat_page_num': best_gt_box[0],
                    'iou': best_iou,
                    'pdf_bbox': pdf_bbox,
                    'cvat_bbox': best_gt_box[1]
                })

        overlaps = [result for result in comparison_results if result['iou'] > 0.5]

        return overlaps, comparison_results

    def calculate_mean_iou(self, comparison_results):
        iou_values = [result['iou'] for result in comparison_results if result['iou'] > 0]
        mean_iou = sum(iou_values) / len(iou_values) if iou_values else 0
        return mean_iou

    def calculate_precision_recall(self, comparison_results, iou_threshold):
        true_positives = sum(1 for result in comparison_results if result['iou'] >= iou_threshold)
        false_positives = len(comparison_results) - true_positives
        false_negatives = len(self.ground_truth_bboxes) - true_positives

        # # Debug prints
        # print(f"IOU Threshold: {iou_threshold}")
        # print(f"True Positives: {true_positives}")
        # print(f"False Positives: {false_positives}")
        # print(f"False Negatives: {false_negatives}")

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        # # Debug prints
        # print(f"Precision: {precision}")
        # print(f"Recall: {recall}")

        return precision, recall

    def calculate_map(self, comparison_results, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
        precision_recall_pairs = [self.calculate_precision_recall(comparison_results, threshold) for threshold in iou_thresholds]
        average_precision = sum(precision for precision, recall in precision_recall_pairs) / len(precision_recall_pairs) if precision_recall_pairs else 0

        # # Debug prints
        # print(f"Precision-Recall Pairs: {precision_recall_pairs}")
        # print(f"Average Precision: {average_precision}")

        return average_precision

