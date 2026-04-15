import cv2
import numpy as np
import os
import ast
import re


# =========================
# Load TXT (Prediction per class)
# =========================
def load_txt(txt_path):
    objects = {}

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        try:
            parts = line.strip().split(" ", 2)
            cls_id = int(parts[0])

            # ambil semua list [ ... ]
            lists = re.findall(r'\[.*?\]', line)

            # segmentation polygon
            seg = ast.literal_eval(lists[1]) if len(lists) > 1 else []

            if cls_id not in objects:
                objects[cls_id] = []

            objects[cls_id].append(seg)

        except:
            continue

    return objects


# =========================
# Convert polygon → mask
# =========================
def polygons_to_instance_masks(polygons, shape):
    masks = []

    for seg in polygons:
        if len(seg) > 0:
            mask = np.zeros(shape[:2], dtype=np.uint8)
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)

    return masks

def iou_dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    iou = intersection / union if union != 0 else 0

    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) != 0 else 0

    return iou, dice

def evaluate_image(gt_mask, pred_objects, img_shape):
    ious = []
    dices = []

    unique_classes = np.unique(gt_mask)

    for gt_cls in unique_classes:
        if gt_cls == 0:
            continue

        pred_cls = gt_cls - 1

        gt_bin = (gt_mask == gt_cls).astype(np.uint8)

        pred_masks = polygons_to_instance_masks(
            pred_objects.get(pred_cls, []),
            img_shape
        )

        if len(pred_masks) == 0:
            ious.append(0)
            dices.append(0)
            continue

        best_iou = 0
        best_dice = 0

        for pm in pred_masks:
            iou, dice = iou_dice(pm, gt_bin)

            if iou > best_iou:
                best_iou = iou
                best_dice = dice

        ious.append(best_iou)
        dices.append(best_dice)

    return ious, dices
# =========================
# Load Ground Truth (multi-class)
# =========================
def load_gt_mask(gt_path):
    return cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)


# =========================
# IoU & Dice per class (with mapping GT->Pred)
# =========================
def compute_metrics_per_class(pred_objects, gt_mask, shape):
    ious = []
    dices = []

    unique_classes = np.unique(gt_mask)

    for gt_cls in unique_classes:
        if gt_cls == 0:
            continue  # skip background

        pred_cls = gt_cls - 1

        # GT binary mask for this class
        gt_bin = (gt_mask == gt_cls).astype(np.uint8)

        # Pred binary mask for mapped class
        polygons = pred_objects.get(pred_cls, [])
        pred_bin = polygons_to_mask(polygons, shape)

        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()

        iou = intersection / union if union != 0 else 0

        dice = (
            (2 * intersection) /
            (pred_bin.sum() + gt_bin.sum())
            if (pred_bin.sum() + gt_bin.sum()) != 0
            else 0
        )

        ious.append(iou)
        dices.append(dice)

        print(f"GT {gt_cls} → Pred {pred_cls} | IoU: {iou:.4f}, Dice: {dice:.4f}")

    return ious, dices


# =========================
# Evaluate Folder
# =========================
def evaluate_folder(pred_txt_dir, gt_dir, image_dir):
    all_ious = []
    all_dices = []

    for file in os.listdir(gt_dir):
        if not file.endswith(".png"):
            continue

        name = os.path.splitext(file)[0]

        gt_path = os.path.join(gt_dir, file)
        txt_path = os.path.join(pred_txt_dir, name + ".txt")
        img_path = os.path.join(image_dir, name + ".jpg")

        if not os.path.exists(txt_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_objects = load_txt(txt_path)

        ious, dices = evaluate_image(gt_mask, pred_objects, img.shape)

        all_ious.extend(ious)
        all_dices.extend(dices)

        print(f"{name} → IoU: {np.mean(ious):.4f}, Dice: {np.mean(dices):.4f}")

    print("\n====================")
    print("FINAL RESULT")
    print("====================")
    print(f"Mean IoU  : {np.mean(all_ious):.4f}")
    print(f"Mean Dice : {np.mean(all_dices):.4f}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    evaluate_folder(
        pred_txt_dir="labels",
        gt_dir="dataset/masks_output",
        image_dir="dataset/images"
    )