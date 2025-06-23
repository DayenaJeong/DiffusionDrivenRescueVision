import os
import csv
import cv2
import random
import re
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from diffusers import StableDiffusionInpaintPipeline
from lpips import LPIPS
from pytorch_fid.fid_score import calculate_fid_given_paths
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import transforms

# ---------------- Configuration ----------------
SEG_MASK_DIR = "datasets/RescueNet/train/train-label-img"
IMAGE_DIR = "datasets/RescueNet/train/train-org-img"
SD_MODEL_PATH = "/home/poseidon/DiffusionDrivenRescueVision/sd15"
YOLO_LABEL_DIR = "datasets/RescueNet_yolo/labels/train"
OUTPUT_ROOT = "outputs/generated_blocked_from_clear"
ROAD_CLEAR_CLASS = 7
TARGET_CLASS_NEW = 8
NUM_GEN_IMAGES = 200
CLIP_TH = 0.80
GUIDANCE_SCALE = 8
STEPS = 80
DEVICE = "cuda"

# ---------------- Paths ----------------
IMAGE_OUT_DIR = Path(OUTPUT_ROOT) / "images"
LABEL_OUT_DIR = Path(OUTPUT_ROOT) / "labels"
PREVIEW_OUT_DIR = Path(OUTPUT_ROOT) / "preview"
CSV_LOG_PATH = Path(OUTPUT_ROOT) / "metadata.csv"

for d in [IMAGE_OUT_DIR, LABEL_OUT_DIR, PREVIEW_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Model Setup ----------------
device = torch.device(DEVICE)

blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    SD_MODEL_PATH, torch_dtype=torch.float32).to(device)
sd_pipe.safety_checker = None

# ---------------- Utility Functions ----------------
def get_images_with_class_present(seg_dir, image_dir, class_index=ROAD_CLEAR_CLASS):
    seg_dir, image_dir = Path(seg_dir), Path(image_dir)
    result = []
    for seg_file in tqdm(seg_dir.glob("*.png"), desc="Scanning masks"):
        seg = cv2.imread(str(seg_file), cv2.IMREAD_GRAYSCALE)
        if seg is not None and np.any(seg == class_index):
            image_file = image_dir / (seg_file.stem.replace("_lab", "") + ".jpg")
            if image_file.exists():
                result.append((image_file, seg_file))
    return result

def randomly_crop_mask(mask: Image.Image, fraction=0.6):
    arr = np.array(mask)
    coords = np.argwhere(arr > 0)
    if len(coords) == 0:
        return mask
    selected = coords[np.random.choice(len(coords), int(len(coords) * fraction), replace=False)]
    new_arr = np.zeros_like(arr)
    for y, x in selected:
        new_arr[y, x] = 255
    return Image.fromarray(new_arr)

def create_mask_from_segmentation(seg_path, image_size=(512, 512), target_class=ROAD_CLEAR_CLASS):
    seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
    if seg is None:
        raise FileNotFoundError(f"Segmentation mask not found: {seg_path}")
    mask = (seg == target_class).astype(np.uint8) * 255
    mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
    return randomly_crop_mask(Image.fromarray(mask), fraction=0.6)

@torch.no_grad()
def clip_sim(gen_img: Image.Image, ref_img: Image.Image) -> float:
    inputs = clip_processor(
        text=["a photo of a blocked disaster road"],
        images=[gen_img, ref_img],
        return_tensors="pt"
    ).to(device)
    outputs = clip_model(**inputs)
    return torch.cosine_similarity(outputs.image_embeds[0], outputs.image_embeds[1], dim=0).item()

def draw_label_preview(image_path: Path, label_path: Path, save_path: Path):
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    draw = ImageDraw.Draw(image)
    w, h = image.size
    with open(label_path) as f:
        for line in f:
            cls, cx, cy, bw, bh = map(float, line.strip().split())
            if int(cls) != TARGET_CLASS_NEW:
                continue
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, max(y1 - 12, 0)), f"Class {int(cls)}", fill="red")
    image.save(save_path)

def save_generated_and_label(gen_img, reference_lbl_path, filename_prefix, prompt, clip_score, csv_writer):
    img_path = IMAGE_OUT_DIR / f"{filename_prefix}.jpg"
    lbl_path = LABEL_OUT_DIR / f"{filename_prefix}.txt"
    preview_path = PREVIEW_OUT_DIR / f"{filename_prefix}.jpg"
    gen_img.save(img_path)
    reference_lbl_path = Path(YOLO_LABEL_DIR) / (filename_prefix.split("_from_")[1] + ".txt")
    with open(reference_lbl_path) as f:
        lines = f.readlines()
    new_lines = [
        line.replace(str(ROAD_CLEAR_CLASS), str(TARGET_CLASS_NEW), 1)
        if int(line.strip().split()[0]) == ROAD_CLEAR_CLASS else line.strip()
        for line in lines
    ]
    with open(lbl_path, "w") as f:
        f.write("\n".join(new_lines))
    draw_label_preview(img_path, lbl_path, preview_path)
    csv_writer.writerow({
        "file": img_path.name,
        "prompt": prompt,
        "clip_score": round(clip_score, 4),
        "ref_image": reference_lbl_path.stem + ".jpg"
    })

# ---------------- Metric Setup ----------------
lpips_model = LPIPS(net='alex').to(device).eval()
to_lpips_tensor = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def lpips_dist(img1, img2):
    t1 = to_lpips_tensor(img1).unsqueeze(0).to(device)
    t2 = to_lpips_tensor(img2).unsqueeze(0).to(device)
    return lpips_model(t1, t2).item()

def run_tsne_visualization(train_img_dir, gen_img_dir, save_path="tsne_result.png", sample_size=100):
    def extract_clip_embedding(img_path):
        image = Image.open(img_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            return clip_model.get_image_features(**inputs).squeeze().cpu().numpy()

    features, labels = [], []
    for img_path in sorted(Path(train_img_dir).glob("*.jpg"))[:sample_size]:
        features.append(extract_clip_embedding(img_path))
        labels.append("train")
    for img_path in sorted(Path(gen_img_dir).glob("*.jpg"))[:sample_size]:
        features.append(extract_clip_embedding(img_path))
        labels.append("generated")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(np.array(features))

    plt.figure(figsize=(10, 7))
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(tsne_result[idxs, 0], tsne_result[idxs, 1], label=label, alpha=0.6)
    plt.legend()
    plt.title("t-SNE of CLIP Embeddings (Train vs Generated)")
    plt.savefig(Path(OUTPUT_ROOT) / save_path)

# ---------------- Main Generation Loop ----------------
def main():
    filtered_imgs = get_images_with_class_present(SEG_MASK_DIR, IMAGE_DIR, class_index=ROAD_CLEAR_CLASS)
    generated, attempts = 0, 0
    max_attempts = 2000

    with open(CSV_LOG_PATH, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file", "prompt", "clip_score", "ref_image"])
        writer.writeheader()

        while generated < NUM_GEN_IMAGES and attempts < max_attempts:
            attempts += 1
            img_path, lbl_path = random.choice(filtered_imgs)

            prompt = random.choice([
                "a disaster-affected road blocked by rubble and fallen trees",
                "an aerial view of a broken road surrounded by debris",
                "a cracked road with rubble from collapsed buildings",
                "a dirt-covered road after a landslide",
                "a road buried in flood debris and branches",
                "a road jammed with wreckage from a collapsed neighborhood",
                "a road partially obstructed by disaster wreckage and dust"
            ])

            try:
                ref_img = Image.open(img_path).convert("RGB").resize((512, 512))
                mask = create_mask_from_segmentation(lbl_path, image_size=(512, 512))
                gen_img = sd_pipe(
                    prompt=prompt,
                    image=ref_img,
                    mask_image=mask,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=STEPS
                ).images[0]

                score = clip_sim(gen_img, ref_img)

                if score >= CLIP_TH:
                    filename_prefix = f"gen_{generated:04d}_from_{img_path.stem}"
                    save_generated_and_label(gen_img, lbl_path, filename_prefix, prompt, score, writer)
                    generated += 1
            except Exception as e:
                continue

    # Post-generation evaluations
    metrics_df = pd.read_csv(CSV_LOG_PATH)
    lpips_scores = []
    for row in tqdm(metrics_df.itertuples(), total=len(metrics_df)):
        gen_img = Image.open(IMAGE_OUT_DIR / row.file).convert("RGB").resize((512, 512))
        ref_img = Image.open(IMAGE_DIR / row.ref_image).convert("RGB").resize((512, 512))
        lpips_scores.append(lpips_dist(gen_img, ref_img))
    metrics_df["lpips"] = lpips_scores
    metrics_df.to_csv(Path(OUTPUT_ROOT) / "metrics_with_lpips.csv", index=False)

    fid = calculate_fid_given_paths([str(IMAGE_DIR), str(IMAGE_OUT_DIR)], batch_size=32, device=device, dims=2048)
    with open(Path(OUTPUT_ROOT) / "fid.txt", "w") as f:
        f.write(f"FID={fid:.2f}\n")

    run_tsne_visualization(IMAGE_DIR, IMAGE_OUT_DIR)

if __name__ == "__main__":
    main()
