import fitz
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from typing import List, Tuple


def coco_to_xyxy(coco_box):
    x, y, w, h = coco_box
    return [x, y, x + w, y + h]

def rescale_bbox(bbox, old_scale, new_scale):
    """
    Rescale a bounding box from old_scale to new_scale.

    Parameters:
    bbox (list): The bounding box to rescale, in [x1, y1, x2, y2] format.
    old_scale (tuple): The old scale, in (width, height) format.
    new_scale (tuple): The new scale, in (width, height) format.

    Returns:
    list: The rescaled bounding box, in [x1, y1, x2, y2] format.
    """
    x1, y1, x2, y2 = bbox
    old_width, old_height = old_scale
    new_width, new_height = new_scale

    x1_new = (x1 / old_width) * new_width
    y1_new = (y1 / old_height) * new_height
    x2_new = (x2 / old_width) * new_width
    y2_new = (y2 / old_height) * new_height

    return [x1_new, y1_new, x2_new, y2_new]

def visualize_bboxes_on_pdf(pdf_path: str, ground_truth_bboxes: List[Tuple[int, List[float]]], detected_bboxes: List[Tuple[int, List[float]]], output_path: str, pages_to_visualize: List[int]):
    doc = fitz.open(pdf_path)
    for page_number in pages_to_visualize:
        page = doc.load_page(page_number - 1)  # Page number correction
        new_scale = page.rect.width, page.rect.height  # Get the width and height of the fitz page

        # Prepare to display
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        plt.figure(figsize=(15, 17))
        plt.imshow(img)
        ax = plt.gca()
        
        # Draw CVAT bounding boxes
        # print(ground_truth_bboxes)
        for bbox in ground_truth_bboxes:
            if bbox[0] == page_number:
                old_scale = bbox[2]  # Get the dimensions of the CVAT image
                x1, y1, x2, y2 = rescale_bbox(bbox[1], old_scale, new_scale)
                rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none', label='CVAT Annotation')
                ax.add_patch(rect)
        
        # Draw detected bounding boxes
        # print(detected_bboxes)
        for bbox in detected_bboxes:
            if bbox[0] == page_number:
                x1, y1, x2, y2 = bbox[1]
                rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='b', facecolor='none', linestyle='dashed', label='Detected Annotation')
                ax.add_patch(rect)
        
        plt.title(f"Page {page_number}")
        plt.axis('off')
        plt.legend(handles=[Rectangle((0, 0), 1, 1, edgecolor='r', facecolor='none', label='CVAT Annotation'),
                            Rectangle((0, 0), 1, 1, edgecolor='b', facecolor='none', linestyle='dashed', label='Detected Annotation')])
        plt.show()

      
def calculate_precision_recall(comparison_results, ground_truth_bboxes, iou_threshold):
    true_positives = sum(1 for result in comparison_results if result['iou'] >= iou_threshold)
    total_detections = len(comparison_results)
    total_annotations = len(ground_truth_bboxes)
    
    false_positives = total_detections - true_positives
    false_negatives = total_annotations - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall

  
def plot_precision_recall_curve(comparison_results, ground_truth_bboxes, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    precisions = []
    recalls = []
    
    for threshold in iou_thresholds:
        precision, recall = calculate_precision_recall(comparison_results, ground_truth_bboxes, threshold)
        precisions.append(precision)
        recalls.append(recall)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iou_thresholds, precisions, label='Precision', marker='o')
    plt.plot(iou_thresholds, recalls, label='Recall', marker='o')
    plt.xlabel('IOU Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
