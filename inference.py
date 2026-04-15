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
parser = argparse.ArgumentParser(description="Edge TPU Video Inference with Save TXT")
parser.add_argument("--model", required=True, help="Path to Edge TPU model (.tflite)")
parser.add_argument("--video", required=True, help="Path to input video")
args = parser.parse_args()

MODEL_PATH = args.model
VIDEO_PATH = args.video


# ==============================
# OUTPUT FOLDER
# ==============================
OUTPUT_DIR = "labels"
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


# ==============================
# OPEN VIDEO
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video file")
    exit()

fps_list = []
frame_count = 0

print("Starting inference...\n")


# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    # DEBUG (aktifkan kalau perlu)
    # for i, out in enumerate(outputs):
    #     print(f"Output {i} shape:", out.shape)

    # ==============================
    # PARSING (SESUAIKAN DENGAN MODELMU)
    # ==============================
    detections = []

    try:
        boxes = outputs[0]
        classes = outputs[1]
        scores = outputs[2]

        masks = outputs[3] if len(outputs) > 3 else None

        num = len(scores)

        for i in range(num):
            score = float(scores[i])

            if score < 0.25:
                continue

            cls = int(classes[i])
            box = boxes[i]

            if masks is not None:
                mask = masks[i]
            else:
                mask = None

            detections.append((cls, score, box, mask))

    except Exception as e:
        print("Error parsing output:", e)
        continue


    # ==============================
    # SAVE TXT
    # ==============================
    frame_name = f"frame_{frame_count:04d}"
    txt_path = os.path.join(OUTPUT_DIR, frame_name + ".txt")

    with open(txt_path, "w") as f:
        for det in detections:
            cls, score, box, mask = det

            # pastikan list biar ada []
            box_list = list(map(float, box))

            if mask is not None:
                mask_array = np.array(mask).flatten()

                # ⚠️ batasi biar file ga besar
                mask_list = list(map(float, mask_array[:200]))
            else:
                mask_list = []

            # FORMAT SESUAI REQUEST
            line = f"{cls} {score} {box_list} {mask_list}"
            f.write(line + "\n")


    # ==============================
    # FPS CALCULATION
    # ==============================
    end_time = time.time()
    inference_time = end_time - start_time
    fps = 1.0 / inference_time

    fps_list.append(fps)
    frame_count += 1

    print(f"Frame {frame_count:04d} | FPS: {fps:.2f}")


cap.release()


# ==============================
# FPS SUMMARY
# ==============================
if len(fps_list) > 0:
    fps_min = min(fps_list)
    fps_avg = sum(fps_list) / len(fps_list)
    fps_max = max(fps_list)

    print("\n===== FPS SUMMARY =====")
    print(f"Total Frames : {frame_count}")
    print(f"FPS Min      : {fps_min:.2f}")
    print(f"FPS Average  : {fps_avg:.2f}")
    print(f"FPS Max      : {fps_max:.2f}")
else:
    print("No frames processed.")