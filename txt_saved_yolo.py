import cv2
import os
import argparse
from ultralytics import YOLO


def save_results_to_txt(results, file_name, output_dir="labels"):
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_name))[0]
    file_path = os.path.join(output_dir, base_name + ".txt")

    with open(file_path, "w") as f:
        for out in results:
            masks = out.masks
            boxes = out.boxes

            for i, box in enumerate(boxes):
                cls_id = int(box.cls.numpy()[0])
                conf = float(box.conf.numpy()[0])
                xyxy = box.xyxy.numpy()[0]

                if masks is not None:
                    seg = masks.xy[i]
                    seg_list = seg.flatten().tolist()
                else:
                    seg_list = []

                f.write(f"{cls_id} {conf} {xyxy.tolist()} {seg_list}\n")


def process_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(
            frame,
            conf=0.5,
            iou=0.3,
            imgsz=256,
            verbose=False,
            stream=True
        )

        save_results_to_txt(results, f"frame_{frame_id}.jpg")
        frame_id += 1

    cap.release()


def process_images(model, input_folder):
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        frame = cv2.imread(img_path)

        results = model.predict(
            frame,
            conf=0.5,
            iou=0.3,
            imgsz=256,
            verbose=False,
            stream=True
        )

        save_results_to_txt(results, img_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True,
                        help="Path ke model (.tflite / .pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="Path ke video file atau folder gambar")
    parser.add_argument("--output", type=str, default="labels",
                        help="Folder output txt")

    args = parser.parse_args()

    model = YOLO(args.model, task="segment")

    if os.path.isdir(args.source):
        process_images(model, args.source)
    else:
        process_video(model, args.source)