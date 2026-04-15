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
parser = argparse.ArgumentParser(description="Edge TPU Image Folder Inference + Save TXT")
parser.add_argument("--model", required=True, help="Path to Edge TPU model (.tflite)")
parser.add_argument("--input", required=True, help="Path to image folder")
args = parser.parse_args()

MODEL_PATH = args.model
INPUT_DIR = args.input


# ==============================
# OUTPUT FOLDER
# ==============================
model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

# buat folder: labels_namamode
OUTPUT_DIR = f"labels_{model_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# LOAD MODEL
# ==============================
print(f"Loading model: {MODEL_PATH}")
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

input_height = input_shape[1]
input_width = input_shape[2]
input_dtype = input_details[0]['dtype']

print("Model loaded successfully.")
print(f"Input size: {input_width}x{input_height}")
print(f"Input dtype: {input_dtype}")

output_details = interpreter.get_output_details()
outputs = [interpreter.get_tensor(o['index']) for o in output_details]
for i, out in enumerate(outputs):
    print(f"Output {i} shape:", out.shape)
# ==============================
# LOAD IMAGE LIST
# ==============================
image_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

print(f"Total images: {len(image_files)}\n")


# ==============================
# MAIN LOOP
# ==============================
fps_list = []

for idx, img_name in enumerate(image_files):
    img_path = os.path.join(INPUT_DIR, img_name)

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Failed to read: {img_name}")
        continue

    start_time = time.time()

    # ==============================
    # PREPROCESS
    # ==============================
    resized = cv2.resize(frame, (input_width, input_height))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    input_tensor = np.expand_dims(resized, axis=0)

    if input_dtype == np.uint8:
        input_tensor = input_tensor.astype(np.uint8)
    else:
        input_tensor = (input_tensor / 255.0).astype(np.float32)

    # ==============================
    # INFERENCE
    # ==============================
    common.set_input(interpreter, input_tensor)
    interpreter.invoke()

    # ==============================
    # GET OUTPUT
    # ==============================
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(o['index']) for o in output_details]

    # ==============================
    # PARSING (SESUAIKAN MODEL)
    # ==============================
    detections = []

    try:
         # ==============================
        # PARSING OUTPUT (SEGMENTATION)
        # ==============================
        output = outputs[0][0]  # (256,256,5)
        mask_class = np.argmax(output, axis=-1)
        mask_conf = np.max(output, axis=-1)
        unique_classes = np.unique(mask_class)

    except Exception as e:
        print("Error parsing output:", e)
        continue


    # ==============================
    # SAVE TXT (NAMA SAMA DENGAN GAMBAR)
    # ==============================
    base_name = os.path.splitext(img_name)[0]
    txt_path = os.path.join(OUTPUT_DIR, base_name + ".txt")

    with open(txt_path, "w") as f:

        for cls in unique_classes:

            # skip background (optional)
            if cls == 0:
                continue

            # ambil mask untuk class ini
            class_mask = (mask_class == cls).astype(np.uint8)

            # confidence rata-rata
            conf = float(mask_conf[class_mask == 1].mean())

            # flatten mask
            mask_flat = class_mask.flatten()

            # batasi biar ga besar
            mask_list = list(map(int, mask_flat[:300]))

            line = f"{cls} {conf} {mask_list}"
            f.write(line + "\n")

    # ==============================
    # FPS
    # ==============================
    end_time = time.time()
    fps = 1.0 / (end_time - start_time)
    fps_list.append(fps)

    print(f"[{idx+1}/{len(image_files)}] {img_name} | FPS: {fps:.2f}")


# ==============================
# FPS SUMMARY
# ==============================
if len(fps_list) > 0:
    print("\n===== FPS SUMMARY =====")
    print(f"Images       : {len(fps_list)}")
    print(f"FPS Min      : {min(fps_list):.2f}")
    print(f"FPS Average  : {sum(fps_list)/len(fps_list):.2f}")
    print(f"FPS Max      : {max(fps_list):.2f}")
else:
    print("No images processed.")