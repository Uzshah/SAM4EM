import os
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchmetrics import JaccardIndex
from medpy import metric
import cv2
# Monkey-patch numpy to replace deprecated np.bool with np.bool_
np.bool = np.bool_

def extract_instances(mask):
    instances = {}
    for instance_id in np.unique(mask):
        if instance_id == 0:
            continue  # Skip background
        instances[instance_id] = (mask == instance_id)
    return instances
    
def semantic_to_instance_mask(semantic_mask):
    """
    Convert a semantic segmentation mask to an instance segmentation mask.

    Arguments:
    semantic_mask -- Tensor or NumPy array of shape (H, W) representing the semantic mask.

    Returns:
    instance_mask -- Tensor or NumPy array of shape (H, W) where each distinct instance has a unique label.
    """
    # Ensure the mask is on CPU and converted to a NumPy array if it's a tensor
    semantic_mask = semantic_mask.squeeze(0)
    if isinstance(semantic_mask, torch.Tensor):
        semantic_mask = semantic_mask.cpu().numpy()

    # Convert to binary mask (1 for object, 0 for background)
    binary_mask = (semantic_mask > 0).astype(np.uint8)
    # print(binary_mask.shape)
    # Perform connected component analysis
    num_labels, instance_mask = cv2.connectedComponents(binary_mask)
    
    # Instance mask will now have unique labels for each connected component
    return torch.from_numpy(instance_mask)
    
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def calculate_precision_recall(gt_mask, pred_mask, iou_threshold=0.75):
    gt_mask = semantic_to_instance_mask(gt_mask)
    pred_mask = semantic_to_instance_mask(pred_mask)
    # Extract instances from masks
    gt_instances = extract_instances(gt_mask)
    pred_instances = extract_instances(pred_mask)

    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    for pred_id, pred_mask in pred_instances.items():
        best_iou = 0
        best_gt_id = -1

        for gt_id, gt_mask in gt_instances.items():
            iou = compute_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id

        if best_iou >= iou_threshold and best_gt_id not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_id)
        else:
            fp += 1

    fn = len(gt_instances) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision

def mask_to_bounding_boxes(mask, expand_factor=1.0):
    # mask = mask.detach().cpu().numpy()
    
    # Ensure the mask is binary
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    
    # Get the dimensions of the mask
    # print(mask.shape)
    c, height, width = mask.shape
    
    # Find contours
    contours, _ = cv2.findContours(mask[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes and normalize them
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Expand the bounding box
        # w_exp = int(w * expand_factor)
        # h_exp = int(h * expand_factor)
        
        # # Adjust x and y to keep the bounding box within the image
        # x_exp = max(0, x - (w_exp - w) // 2)
        # y_exp = max(0, y - (h_exp - h) // 2)
        
        # # Ensure the bounding box does not exceed the image boundaries
        # if x_exp + w_exp > width:
        #     w_exp = width - x_exp
        # if y_exp + h_exp > height:
        #     h_exp = height - y_exp
        
        # x_norm = x_exp / width
        # y_norm = y_exp / height
        # w_norm = w_exp / width
        # h_norm = h_exp / height
        
        bounding_boxes.append([x, y, w, h])
    
    return np.array(bounding_boxes)

# import numpy as np

def calculate_iou_boxes(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Boxes are defined by [x_min, y_min, x_max, y_max].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    iou = intersection / union if union != 0 else 0

    return iou

def calculate_precision_at_75(pred_boxes, gt_boxes):
    """
    Calculate Precision at 75% IoU for bounding boxes.
    """
    tp = 0  # True positives
    fp = 0  # False positives
    matched_gt = set()

    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue

            iou = calculate_iou_boxes(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= 0.75:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return precision

def filter_small_boxes(boxes, area_threshold):
    """
    Filter out bounding boxes that are smaller than a given area threshold.
    """
    filtered_boxes = []
    for box in boxes:
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area >= area_threshold:
            filtered_boxes.append(box)
    return filtered_boxes


def compute_mAP(y_true, y_pred, num_classes=2, thresholds=np.arange(0.0, 1.1, 0.1)):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    # gt_boxes, pred_boxes = mask_to_bounding_boxes(y_true), mask_to_bounding_boxes(y_pred[0])
    # gt_boxes, pred_boxes = filter_small_boxes(gt_boxes, 0), filter_small_boxes(pred_boxes, 20000)
    # print(f"len of gt boxes {len(gt_boxes)} and len of pred boxes {len(pred_boxes)}")
    # AP75 = calculate_precision_at_75(pred_boxes, gt_boxes)
    # print(y_true.shape, y_pred.shape)
    AP75 = calculate_precision_recall(y_true, y_pred.squeeze(1))
    # print(AP75)
    aps= []
    ap_at_thresholds = {}

    for cls in range(1, num_classes):
        precisions = []
        recalls = []

        for threshold in thresholds:
            y_true_cls = (y_true == cls).astype(float)
            y_pred_cls = (y_pred == cls).astype(float)
            
            prec, recall = confusion(y_true_cls, (y_pred_cls > threshold).astype(float))
            precisions.append(prec)
            recalls.append(recall)
        
        # Sort by recall and compute interpolated precision
        precisions = np.array(precisions)
        recalls = np.array(recalls)

        sorted_indices = np.argsort(recalls)
        precisions = precisions[sorted_indices]
        recalls = recalls[sorted_indices]
        
        interpolated_precisions = []
        for recall_level in np.arange(0.0, 1.1, 0.1):
            if np.sum(recalls >= recall_level) == 0:
                interpolated_precisions.append(0.0)
            else:
                interpolated_precisions.append(np.max(precisions[recalls >= recall_level]))
        
        ap = np.mean(interpolated_precisions)
        aps.append(ap)
        
        for threshold in [0.25, 0.5]:
            recall_at_threshold = interpolated_precisions[int(threshold * 10)]
            if f"mAP{int(threshold * 100)}" not in ap_at_thresholds:
                ap_at_thresholds[f"mAP{int(threshold * 100)}"] = []
            ap_at_thresholds[f"mAP{int(threshold * 100)}"].append(recall_at_threshold)
    
    mAP = np.mean(aps)
    for key in ap_at_thresholds:
        ap_at_thresholds[key] = np.mean(ap_at_thresholds[key])
    
    # Return mAP and AP at specific thresholds
    return float(mAP),ap_at_thresholds.get("mAP25", 0.0),ap_at_thresholds.get("mAP50", 0.0), AP75
    

def confusion(y_true, y_pred):
    """ A placeholder confusion function that calculates precision and recall.
    This function should be replaced with your actual implementation of precision and recall calculation.
    """
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))

    prec = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return prec, recall



# def mIoU(mask, pred_mask, smooth=1e-10, n_classes=2):
    
    
#     clas =1
#     true_class = pred_mask == clas
#     true_label = mask == clas

#     if true_label.long().sum().item() == 0: #no exist label in this loop
#         iou = np.nan
#     else:
#         intersect = torch.logical_and(true_class, true_label).sum().float().item()
#         union = torch.logical_or(true_class, true_label).sum().float().item()

#         iou = (intersect + smooth) / (union +smooth)
#             # iou_per_class.append(iou)
#             # print(iou_per_class)
#     return torch.tensor(iou, dtype=torch.float32, device=mask.device)

# import torch

def calculate_iou(gt_mask, pred_mask, c):
    """ Calculate IoU for a specific class c (ignoring background class 0).
    
    Args:
    - gt_mask: Ground truth mask tensor (shape: [H, W])
    - pred_mask: Predicted mask tensor (shape: [H, W])
    - c: Class label for which IoU is calculated
    
    Returns:
    - IoU score for class c (float)
    """
    # # Convert tensors to numpy arrays if they are not already
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.numpy()
    
    # # Convert to binary masks for class c
    gt_c = (gt_mask == c).astype(np.float32)
    pred_c = (pred_mask == c).astype(np.float32)
    
    # # Compute intersection and union
    # intersection = np.logical_and(gt_c, pred_c).sum()
    # union = np.logical_or(gt_c, pred_c).sum()
    
    # # Avoid division by zero
    # iou = intersection / (union + 1e-10)
    if pred_c.sum() > 0 and gt_c.sum() > 0:
        iou = metric.binary.jc(pred_c, gt_c)
        dc = metric.binary.dc(pred_c, gt_c)
        # print(iou, dc)
    else:
        iou, dc = 0.0, 0.0
    return iou,dc

def mIoU(gt_masks, pred_masks, num_classes=2):
    """ Calculate mean IoU for a list of ground truth masks and predicted masks, ignoring background class (class 0).
    
    Args:
    - gt_masks: List or array of ground truth binary masks (each mask is a tensor of shape [H, W])
    - pred_masks: List or array of predicted binary masks (each mask is a tensor of shape [H, W])
    - num_classes: Total number of classes including background
    
    Returns:
    - Mean IoU excluding background class (float)
    """
    assert len(gt_masks) == len(pred_masks), "Number of ground truth masks and predicted masks must be the same."
    
    iou_scores = []
    dc_scores = []
    num_classes = gt_masks.max()+1
    # print(gt_masks.shape, num_classes)
    for i in range(len(gt_masks)):
        for c in range(1, 2):  # Skip class 0 (background)
            iou,dc = calculate_iou(gt_masks[i], pred_masks[i], c)
            # print(iou, dc)
            # if iou is None or iou<0:
            #     iou_scores.append(iou)
                
            iou_scores.append(iou)
            dc_scores.append(dc)
    
    mean_iou = np.mean(iou_scores)
    mean_dc = np.mean(dc_scores)
    
    return mean_iou, mean_dc


# 
######################AP###############################################
def calculate_ap(gt_masks, pred_masks, num_classes, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """ Calculate Average Precision (AP) for semantic segmentation.
    
    Args:
    - gt_masks: List of ground truth binary masks (each mask is a numpy array of shape [H, W])
    - pred_masks: List of predicted binary masks (each mask is a numpy array of shape [H, W])
    - num_classes: Number of classes (excluding background)
    - iou_thresholds: List or array of IoU thresholds
    
    Returns:
    - AP for each class (numpy array of shape [num_classes])
    """
    assert len(gt_masks) == len(pred_masks), "Number of ground truth and predicted masks must be the same."
    
    ap_scores = np.zeros(num_classes)
    
    for c in range(1, num_classes + 1):  # Iterate over classes (1 to num_classes)
        precisions = []
        recalls = []
        
        for threshold in iou_thresholds:
            tp = 0  # True positive
            fp = 0  # False positive
            fn = 0  # False negative
            
            for gt_mask, pred_mask in zip(gt_masks, pred_masks):
                gt_c = (gt_mask == c).astype(np.float32)
                pred_c = (pred_mask == c).astype(np.float32)
                
                iou = calculate_iou(gt_c, pred_c)
                
                if iou >= threshold:
                    tp += 1
                else:
                    fp += 1
            
            fn = len(gt_masks) - tp  # All missed ground truths are false negatives
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using precision-recall curve
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        # Compute AP using trapezoidal rule
        ap = np.trapz(precisions, recalls)
        
        ap_scores[c-1] = ap
    
    return ap_scores

# iou_metric = JaccardIndex(task="binary", num_classes= 1, ignore_index=0)

def pixel_accuracy(mask, output):
    with torch.no_grad():
        # output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
        accuracy = torch.tensor(accuracy, dtype=torch.float32, device=mask.device)
    return accuracy


def dice_coefficient(y_true, y_pred):
    smooth = 0.1
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (
                torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score

def compute_semantic_metrics(gt_semantic, pred_semantic):
    miou, dice = mIoU(gt_semantic, pred_semantic)
    
    acc = pixel_accuracy(gt_semantic, pred_semantic)
    # dice = dice_coefficient(gt_semantic, pred_semantic)
    # mAP, mAP25, mAP50, mAP75 = compute_mAP(gt_semantic, pred_semantic)
    mAP, mAP25, mAP50, mAP75 = 0,0,0, 0#calculate_precision_recall(gt_semantic.detach().cpu().numpy(), 
                                                                # pred_semantic.squeeze(1).detach().cpu().numpy())
    return [miou, acc, dice, mAP, mAP25, mAP50, mAP75 ]
    
# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


class Evaluator(object):

    def __init__(self, median_align=True):

        self.median_align = median_align
        # depthor and Accuracy metric trackers
        self.metrics = {}
        self.metrics['ss/miou'] = AverageMeter()
        self.metrics['ss/acc'] = AverageMeter()
        self.metrics['ss/dice'] = AverageMeter()
        self.metrics['ss/map'] = AverageMeter()
        self.metrics['ss/map25'] = AverageMeter()
        self.metrics['ss/map50'] = AverageMeter()
        self.metrics['ss/map75'] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        # depth
        self.metrics['ss/miou'].reset()
        self.metrics['ss/acc'].reset()
        self.metrics['ss/dice'].reset()
        self.metrics['ss/map'].reset()
        self.metrics['ss/map25'].reset()
        self.metrics['ss/map50'].reset()
        self.metrics['ss/map75'].reset()
        
    def compute_eval_metrics(self, gt_semantic=None,
                             pred_semantic=None):
        """
        Computes metrics used to evaluate the model
        """
        
        N = gt_semantic.shape[0]
        ##Semantic 
        miou, acc, dice, map, map25, map50, map75 = compute_semantic_metrics(gt_semantic, pred_semantic)
        self.metrics['ss/miou'].update(miou, N)
        self.metrics['ss/acc'].update(acc, N)
        self.metrics['ss/dice'].update(dice, N)
        self.metrics['ss/map'].update(map, N)
        self.metrics['ss/map25'].update(map25, N)
        self.metrics['ss/map50'].update(map50, N)
        self.metrics['ss/map75'].update(map75, N)
        
    def return_metrics(self):
        avg_metrics = []
        avg_metrics.append(self.metrics["ss/miou"].avg)
        avg_metrics.append(self.metrics["ss/acc"].avg)
        avg_metrics.append(self.metrics["ss/dice"].avg)
        avg_metrics.append(self.metrics["ss/map"].avg)
        avg_metrics.append(self.metrics["ss/map25"].avg)
        avg_metrics.append(self.metrics["ss/map50"].avg)
        avg_metrics.append(self.metrics["ss/map75"].avg)
        return avg_metrics
        
    def print(self, dir=None):
        avg_metrics = []
        avg_metrics.append(self.metrics["ss/miou"].avg)
        avg_metrics.append(self.metrics["ss/acc"].avg)
        avg_metrics.append(self.metrics["ss/dice"].avg)
        avg_metrics.append(self.metrics["ss/map"].avg)
        avg_metrics.append(self.metrics["ss/map25"].avg)
        avg_metrics.append(self.metrics["ss/map50"].avg)
        avg_metrics.append(self.metrics["ss/map75"].avg)

        print("\n********************Semantic*******************************")
        print("\n  "+ ("{:>9} | " * 7).format("miou", "acc", "dice", "mAP", "mAP25", "mAP50", "mAP75"))
        print(("&  {: 8.5f} " * 7).format(*avg_metrics))