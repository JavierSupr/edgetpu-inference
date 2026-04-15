import cv2
import numpy as np


def colorize_mask(gt):
    h, w = gt.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 🎨 mapping warna (BGR)
    COLOR_MAP = {
        0: (0, 0, 0),        # background
        1: (0, 255, 255),    # kuning
        2: (0, 255, 0),      # hijau
        3: (0, 0, 255),      # merah
        4: (255, 0, 0),      # biru
    }

    for pixel_val, color in COLOR_MAP.items():
        color_mask[gt == pixel_val] = color

    return color_mask


if __name__ == "__main__":
    gt_path = r"C:\Users\javie\Documents\Penelitian\Program\Inference Coral\dataset\masks_output\Desain-tanpa-judul-9-_mp4-0045_jpg.rf.102691951a3b6c8a658035a35843d73e.png"

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    print("Unique pixel:", np.unique(gt))

    color_mask = colorize_mask(gt)

    cv2.imshow("Original GT", gt )  # biar kelihatan
    cv2.imshow("Colorized GT", color_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()