# SGGA-for-Disaster-Detection

**SGGA (Semantic-Guided Generative Augmentation)** is a diffusion-based data augmentation framework designed to address rare-class imbalance in disaster object detection. It synthesizes *Road-Blocked* scenes from *Road-Clear* UAV imagery using semantic masks, prompt-based inpainting, and quality-controlled filtering.

> 🛰️ This work was conducted as part of a defense-industry collaboration and is currently under review at the International Journal of Control, Automation, and Systems (2025).

---

## 📁 Directory Structure

```
SGGA-for-Disaster-Detection/
├── scripts/
│   ├── diffusion/
│   │   ├── sgga_generate_blocked.py     # Main SGGA pipeline
│   │   ├── train_lora_rescuenet.py      # LoRA finetuning on RescueNet
│   │   └── utils_diffusion.py           # FID, LPIPS, t-SNE utilities
│   ├── check_yolo_labels.py             # Visualize YOLO label boxes
│   ├── convert_mask_to_yolo.py          # Convert segmentation masks to YOLO labels
│   └── organize_yolo_split.py           # Train/val/test YOLO dataset splitter
├── datasets/
│   ├── RescueNet/                       # Raw segmentation and RGB images
│   └── RescueNet_yolo/                 # Converted YOLO-style labels
├── outputs/
│   └── generated_blocked_from_clear/   # Results (auto-generated)
└── README.md
```

---

## ✨ Key Features

* **Mask-guided Stable Diffusion**: Generate "Road-Blocked" images from real "Road-Clear" UAV imagery using semantic segmentation.
* **Prompt diversity**: Uses carefully curated disaster prompts (e.g., “a road blocked by collapsed buildings”).
* **Semantic & perceptual filtering**: Filters using CLIP similarity, LPIPS, and FID to retain only high-quality generations.
* **YOLO-compatible outputs**: Converts synthetic samples into bounding-box `.txt` files for downstream object detection.
* **Visual diagnostics**: Generates preview images, metadata logs, and t-SNE embedding maps.

---

## ⚙️ Installation

```bash
git clone https://github.com/DayenaJeong/SGGA-for-Disaster-Detection.git
cd SGGA-for-Disaster-Detection
pip install -r requirements.txt
```

You will also need the following Hugging Face models downloaded locally:

* `runwayml/stable-diffusion-inpainting`
* `Salesforce/blip2-opt-2.7b`
* `openai/clip-vit-base-patch32`

Set the correct path for `SD_MODEL_PATH` in the scripts.

---

## ▶️ How to Use

### 1. Generate Road-Blocked Images

```bash
python scripts/diffusion/sgga_generate_blocked.py
```

This script:

* Extracts “Road-Clear” regions from segmentation masks.
* Samples random prompts and inpaints damage using Stable Diffusion.
* Filters results based on CLIP score, LPIPS threshold, and mean pixel intensity.
* Outputs are saved to `outputs/generated_blocked_from_clear/`

### 2. Optional: Train Diffusion with LoRA

```bash
python scripts/diffusion/train_lora_rescuenet.py
```

Finetunes the inpainting model with domain-specific prompts and produces an alternative set of augmented images for comparison.

---

## 🧪 Evaluation Metrics

* **CLIP Similarity** ≥ 0.80 (semantic correctness)
* **LPIPS** ≤ 0.68 (visual realism)
* **FID** (distributional similarity)
* **t-SNE** visualization of CLIP embeddings

All results are automatically stored under `outputs/generated_blocked_from_clear/`.

---

## 📦 YOLO Dataset Tools

* `convert_mask_to_yolo.py`: Converts segmentation labels into YOLO format
* `organize_yolo_split.py`: Splits YOLO dataset into train/val/test
* `check_yolo_labels.py`: Visualizes bounding boxes for quality control

---

## 📊 Example Output Folder

```
outputs/generated_blocked_from_clear/
├── images/        ← Synthetic images
├── labels/        ← YOLO format .txt labels
├── preview/       ← Visualization with bounding boxes
├── metadata.csv   ← Prompt and score logs
├── metrics_with_lpips.csv
├── fid.txt
└── tsne_result.png
```

---

## 🧩 Citation

If you use SGGA in your research, please cite:

```bibtex
@article{jeong2025sgga,
  title={SGGA: Semantic-Guided Generative Augmentation for Object Detection in Highly Imbalanced Disaster Imagery},
  author={Jeong, Dayena and Heo, Dongwook and Ahn, Seonghyeok and Choi, Jonggeun and Choi, Sunglok},
  journal={International Journal of Control, Automation, and Systems},
  year={2025}
}
```

---

## 🙋‍♀️ Author

**Dayena Jeong**
M.S. Student @ SEOULTECH, Defense Applied AI Lab
📫 [LinkedIn](https://linkedin.com/in/dayenajeong) · 📧 [pasteldiana@seoultech.ac.kr](mailto:pasteldiana@seoultech.ac.kr)

---
