import shutil
from pathlib import Path

def organize_from_split_with_maskpaths(
    image_root, mask_root, label_dir, split_file, split_type, out_root
):
    """
    Organize images and labels using val.txt which contains both .jpg and .png relative paths.

    Args:
        image_root (Path): Path to Rellis-3D images directory
        mask_root (Path): Path to Rellis-3D masks directory
        label_dir (Path): Path to YOLO .txt label files (flat)
        split_file (Path): Path to val.txt or train.txt (with two columns)
        split_type (str): 'train' or 'val'
        out_root (Path): Output root path (e.g., data/rellis3d/)
    """
    image_root = Path(image_root)
    mask_root = Path(mask_root)
    label_dir = Path(label_dir)
    split_file = Path(split_file)
    out_img = Path(out_root) / "images" / split_type
    out_lbl = Path(out_root) / "labels" / split_type
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    moved = 0
    with open(split_file) as f:
        for line in f:
            jpg_rel, png_rel = line.strip().split()
            stem = Path(png_rel).stem  # label stem: same as .png/.txt name

            image_path = image_root / jpg_rel
            label_path = label_dir / f"{stem}.txt"

            if image_path.exists() and label_path.exists():
                shutil.copy(image_path, out_img / image_path.name)
                shutil.copy(label_path, out_lbl / label_path.name)
                moved += 1
            else:
                print(f"[Warning] Missing file - img: {image_path.exists()}, lbl: {label_path.exists()} - {stem}")

    print(f"[Done] {split_type} set organized with {moved} samples.")

# 예시 실행용
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    organize_from_split_with_maskpaths(
        image_root=BASE / "data/rellis3d/images/Rellis-3D",
        mask_root=BASE / "data/rellis3d/masks/Rellis-3D",
        label_dir=BASE / "data/rellis3d/labels",
        split_file=BASE / "data/rellis3d/splits/train.txt",
        split_type="train",
        out_root=BASE / "data/rellis3d"
    )

    organize_from_split_with_maskpaths(
        image_root=BASE / "data/rellis3d/images/Rellis-3D",
        mask_root=BASE / "data/rellis3d/masks/Rellis-3D",
        label_dir=BASE / "data/rellis3d/labels",
        split_file=BASE / "data/rellis3d/splits/val.txt",
        split_type="val",
        out_root=BASE / "data/rellis3d"
    )