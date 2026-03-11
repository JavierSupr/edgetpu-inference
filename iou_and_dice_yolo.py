import cv2
import argparse
import numpy as np
import os

from ultralytics import YOLO


# ==============================
# ARGUMENT PARSER
# ==============================
parser = argparse.ArgumentParser(description="YOLO Segmentation Dataset Evaluation")
parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
parser.add_argument("--test_dir", required=True, help="Path to test dataset folder")
args = parser.parse_args()

MODEL_PATH = args.model
TEST_DIR = args.test_dir
CONF_THRES = 0.7
IOU_THRES = 0.2
IMG_SIZE = 256


# ==============================
# LOAD MODEL
# ==============================
print(f"\nLoading YOLO model: {MODEL_PATH}")
model = YOLO(MODEL_PATH, task='segment')

print("Model loaded successfully.")


# ==============================
# METRIC FUNCTION
# ==============================
def compute_metrics(pred_mask, true_mask, num_classes):

    iou_per_class = {}
    dice_per_class = {}

    for cls in range(num_classes):

        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)

        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()

        if union == 0:
            iou = 0
        else:
            iou = intersection / (union + 1e-7)

        denom = pred_cls.sum() + true_cls.sum()

        if denom == 0:
            dice = 0
        else:
            dice = (2 * intersection) / (denom + 1e-7)

        iou_per_class[cls] = iou
        dice_per_class[cls] = dice

    mean_iou = np.mean(list(iou_per_class.values()))
    mean_dice = np.mean(list(dice_per_class.values()))

    return mean_iou, mean_dice, iou_per_class, dice_per_class


# ==============================
# YOLO -> SEMANTIC MASK
# ==============================
def build_semantic_mask(result, shape):

    h, w = shape[:2]

    semantic_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return semantic_mask

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    # resize semua mask dulu
    processed_masks = []
    for mask in masks:
        mask = cv2.resize(mask, (w, h))
        mask = mask > 0.5
        processed_masks.append(mask)

    processed_masks = np.array(processed_masks)

    # =============================
    # STEP 1 : gambar kelas 3 & 4 dulu
    # =============================
    for i, cls in enumerate(classes):

        if cls in [0, 1]:

            semantic_mask[processed_masks[i]] = cls

    # =============================
    # STEP 2 : kelas 1 & 2 hanya di area kosong
    # =============================
    for i, cls in enumerate(classes):

        if cls in [2, 3]:

            mask = processed_masks[i]

            # hanya tempat yang belum diisi
            mask = np.logical_and(mask, semantic_mask == 0)

            semantic_mask[mask] = cls

    return semantic_mask


# ==============================
# DATASET EVALUATION
# ==============================
def evaluate_dataset(test_dir):

    images_dir = os.path.join(test_dir, "images")
    masks_dir = os.path.join(test_dir, "masks")

    image_files = sorted(os.listdir(images_dir))

    total_iou = []
    total_dice = []

    print("\nStarting dataset evaluation...\n")

    for idx, image_file in enumerate(image_files):

        base_name = os.path.splitext(image_file)[0]

        image_path = os.path.join(images_dir, image_file)

        possible_mask_extensions = [".png", ".jpg", ".jpeg"]

        mask_path = None
        for ext in possible_mask_extensions:
            candidate = os.path.join(masks_dir, base_name + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            print(f"Mask not found for {image_file}, skipping...")
            continue
        # ==============================
        # LOAD IMAGE
        # ==============================
        image = cv2.imread(image_path)
        true_mask = cv2.imread(mask_path, 0)

        # resize sama seperti inference pipeline
        image_resized = cv2.resize(image, (256, 256))
        true_mask = cv2.resize(true_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # ==============================
        # YOLO INFERENCE (SAMA SEPERTI CORAL PIPELINE)
        # ==============================
        results = model.predict(image_resized, conf=0.2, iou=0.2, imgsz=256, verbose=False)
        result = results[0]
        print(result)

        pred_mask = build_semantic_mask(result, image_resized.shape)

        pred_mask = build_semantic_mask(result, image.shape)

        num_classes = int(np.max(true_mask)) + 1

        iou, dice, iou_per_class, dice_per_class = compute_metrics(pred_mask, true_mask, num_classes)

        total_iou.append(iou)
        total_dice.append(dice)
        print(f"\nImage: {image_file}")

        for cls in range(num_classes):
            print(f"Class {cls} | IoU: {iou_per_class[cls]:.4f} | Dice: {dice_per_class[cls]:.4f}")

            print(f"Mean IoU  : {iou:.4f}")
            print(f"Mean Dice : {dice:.4f}")
            print("-" * 40)

        print(f"[{idx+1}/{len(image_files)}] IoU: {iou:.4f} | Dice: {dice:.4f}")

    print("\n===== DATASET RESULT =====")
    print(f"Images Evaluated : {len(total_iou)}")
    print(f"Mean IoU         : {np.mean(total_iou):.4f}")
    print(f"Mean Dice        : {np.mean(total_dice):.4f}")
    print("=================================\n")


# ==============================
# RUN
# ==============================
evaluate_dataset(TEST_DIR)