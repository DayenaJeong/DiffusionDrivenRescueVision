import cv2
import numpy as np
from pathlib import Path

def convert_mask_to_yolo_bbox(mask_path, output_path, class_ids=None, min_area=100):
    """
    Convert a semantic segmentation mask image to YOLO format bounding boxes.

    Args:
        mask_path (Path): Path to the input .png segmentation mask.
        output_path (Path): Path to the output YOLO .txt label file.
        class_ids (list[int], optional): List of class IDs to include (default: all except background).
        min_area (int): Minimum area of a contour to consider (to filter noise).
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[Warning] Cannot read mask image: {mask_path}")
        return

    h, w = mask.shape
    labels = []

    for class_id in np.unique(mask):
        if class_id == 0:
            continue  # skip background
        if class_ids and class_id not in class_ids:
            continue

        bin_mask = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    with open(output_path, "w") as f:
        f.write("\n".join(labels))

def batch_convert_masks_to_yolo(mask_dir, label_dir, class_ids=None):
    """
    Recursively convert all .png mask files in a directory to YOLO format labels.

    Args:
        mask_dir (str or Path): Directory containing mask .png files (can include subdirectories).
        label_dir (str or Path): Output directory for YOLO .txt labels.
        class_ids (list[int], optional): Class IDs to include (default: all except background).
    """
    mask_dir = Path(mask_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = list(mask_dir.glob("**/*.png"))
    print(f"[INFO] Found {len(mask_paths)} mask files.")

    for mask_path in mask_paths:
        output_path = label_dir / f"{mask_path.stem}.txt"
        convert_mask_to_yolo_bbox(mask_path, output_path, class_ids)

    print("[Done] Mask to YOLO conversion complete.")

batch_convert_masks_to_yolo(
    "data/rellis3d/masks/Rellis-3D",
    "data/rellis3d/labels",
    class_ids=range(1, 16)
)