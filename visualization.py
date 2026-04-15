import cv2
import numpy as np
import os
import re
import ast

CLASS_COLORS = {
    0: (0, 255, 255),  # Kuning
    1: (0, 255, 0),    # Hijau
    2: (0, 0, 255),    # Merah
    3: (255, 0, 0)     # Biru
}

def load_txt(txt_path):
    objects = []

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        try:
            # Ambil class & confidence
            first_split = line.strip().split(" ", 2)
            cls_id = int(first_split[0])
            conf = float(first_split[1])

            # Ambil semua list [ ... ]
            lists = re.findall(r'\[.*?\]', line)

            bbox = ast.literal_eval(lists[0])
            seg = ast.literal_eval(lists[1]) if len(lists) > 1 else []

            objects.append({
                "class": cls_id,
                "conf": conf,
                "bbox": bbox,
                "seg": seg
            })

        except Exception as e:
            print("Error parsing line:", line)
            print(e)

    return objects


def visualize(image_path, txt_path):
    img = cv2.imread(image_path)

    objects = load_txt(txt_path)

    for obj in objects:
        if obj["class"] != 3:
            continue
        bbox = obj["bbox"]
        seg = obj["seg"]

        # 🎨 Warna random
        color = CLASS_COLORS.get(obj["class"], (255, 255, 255))

        # =====================
        # Draw Mask (Polygon)
        # =====================
        if len(seg) > 0:
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)

            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # =====================
        # Draw Bounding Box
        # =====================
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # =====================
        # Label
        # =====================
        label = f"{obj['class']} {obj['conf']:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


if __name__ == "__main__":
    image_path = r"dataset\images\Desain-tanpa-judul-9-_mp4-0045_jpg.rf.102691951a3b6c8a658035a35843d73e.jpg"
    txt_path = r"labels\Desain-tanpa-judul-9-_mp4-0045_jpg.rf.102691951a3b6c8a658035a35843d73e.txt"

    result = visualize(image_path, txt_path)

    cv2.imshow("Mask Visualization", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()