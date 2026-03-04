import cv2
import time
import argparse
import numpy as np
import os

from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common


# ==============================
# ARGUMENT PARSER
# ==============================
parser = argparse.ArgumentParser(description="Edge TPU Video + Dataset Evaluation")
parser.add_argument("--model", required=True, help="Path to Edge TPU model (.tflite)")
parser.add_argument("--video", required=True, help="Path to input video")
parser.add_argument("--test_dir", required=True, help="Path to test dataset folder")
args = parser.parse_args()

MODEL_PATH = args.model
VIDEO_PATH = args.video
TEST_DIR = args.test_dir


# ==============================
# LOAD MODEL
# ==============================
print(f"\nLoading model: {MODEL_PATH}")
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]
input_dtype = input_details[0]['dtype']

print("Model loaded successfully.")
print(f"Input size  : {input_width}x{input_height}")
print(f"Input dtype : {input_dtype}")


# ==============================
# METRIC FUNCTION
# ==============================
def compute_metrics(pred_mask, true_mask, num_classes):
    iou_list = []
    dice_list = []

    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)

        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()

        if union == 0:
            continue

        iou = intersection / (union + 1e-7)
        dice = (2 * intersection) / (pred_cls.sum() + true_cls.sum() + 1e-7)

        iou_list.append(iou)
        dice_list.append(dice)

    if len(iou_list) == 0:
        return 0, 0

    return np.mean(iou_list), np.mean(dice_list)


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

        # ambil nama tanpa extension
        base_name = os.path.splitext(image_file)[0]

        image_path = os.path.join(images_dir, image_file)

        # cari mask dengan ekstensi berbeda
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

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)  # grayscale

        # ---------------- Preprocess ----------------
        resized = cv2.resize(image, (input_width, input_height))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        input_tensor = np.expand_dims(resized, axis=0)

        if input_dtype == np.uint8:
            input_tensor = input_tensor.astype(np.uint8)
        else:
            input_tensor = (input_tensor / 255.0).astype(np.float32)

        # ---------------- Inference ----------------
        common.set_input(interpreter, input_tensor)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])

        if len(output_data.shape) == 4:
            pred_mask = np.argmax(output_data[0], axis=-1)
            num_classes = output_data.shape[-1]
        else:
            pred_mask = output_data[0]
            num_classes = int(np.max(mask)) + 1

        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        iou, dice = compute_metrics(pred_mask, mask, num_classes)

        total_iou.append(iou)
        total_dice.append(dice)

        print(f"[{idx+1}/{len(image_files)}] IoU: {iou:.4f} | Dice: {dice:.4f}")

    print("\n===== DATASET RESULT =====")
    print(f"Images Evaluated : {len(total_iou)}")
    print(f"Mean IoU         : {np.mean(total_iou):.4f}")
    print(f"Mean Dice        : {np.mean(total_dice):.4f}")
    print("=================================\n")


# ==============================
# RUN DATASET EVALUATION
# ==============================
evaluate_dataset(TEST_DIR)