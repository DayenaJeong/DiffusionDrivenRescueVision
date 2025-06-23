import os, random, json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    CLIPProcessor, CLIPModel, CLIPTokenizer,
    Blip2Processor, Blip2ForConditionalGeneration
)
from diffusers import StableDiffusionPipeline, DDPMScheduler
from accelerate import Accelerator
from lpips import LPIPS
from pytorch_fid.fid_score import calculate_fid_given_paths
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

# -------------------- Configuration --------------------
SD_MODEL_PATH   = "/home/poseidon/DiffusionDrivenRescueVision/sd15"
TRAIN_IMG_DIR   = "datasets/RescueNet_yolo/images/train"
PROMPT_FILE     = "scripts/diffusion/prompts.txt"
LORA_OUTPUT_DIR = "outputs/diffusion/lora_output"
FINAL_GEN_DIR   = "outputs/lora_output_images_only_clip3"
BATCH_SIZE      = 1
EPOCHS          = 0
GUIDANCE_SCALE  = 8
STEPS           = 80
IMAGE_SIZE      = 512
PATIENCE        = 3
NUM_GEN_IMAGES  = 200
CLIP_TH         = 0.80
LPIPS_TH        = 0.65

torch.manual_seed(42)
random.seed(42)
device = "cuda"

os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_GEN_DIR, exist_ok=True)

# -------------------- Prompt Cleaning --------------------
def clean_caption(caption: str) -> str:
    caption = caption.lower()
    tokens = caption.split()
    caption = " ".join([tokens[i] for i in range(len(tokens)) if i == 0 or tokens[i] != tokens[i - 1]])

    replacements = {
        "a lot of debris": "scattered debris",
        "a bunch of debris": "scattered debris",
        "many debris": "scattered debris",
        "lots of debris": "scattered debris",
        "a lot of trees": "many trees"
    }
    for k, v in replacements.items():
        caption = caption.replace(k, v)

    banned_phrases = ["a lot of debris", "many debris", "a bunch of trees", "lots of stuff", "something"]
    if any(p in caption for p in banned_phrases):
        return None

    required_keywords = ["debris", "road", "tree"]
    if not any(k in caption for k in required_keywords):
        return None

    excluded_keywords = ["bird", "beach", "sky", "sunset", "cloud", "sea"]
    if any(k in caption for k in excluded_keywords):
        return None

    if len(caption.split()) < 4:
        return None

    return caption.strip()

# -------------------- BLIP Caption Generator --------------------
def generate_prompts_with_blip(image_dir: str, output_file: str):
    print("[BLIP] Generating captions...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda")

    image_dir = Path(image_dir)
    lines = []
    for img_path in tqdm(sorted(image_dir.glob("*.jpg"))):
        image = Image.open(img_path).convert("RGB")
        pixel_values = processor.image_processor(images=image, return_tensors="pt")["pixel_values"].to("cuda")

        with torch.no_grad():
            outputs = model.generate(pixel_values=pixel_values, max_new_tokens=30)
            caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

        cleaned = clean_caption(caption)
        if cleaned:
            lines.append(f"{img_path.name}|{cleaned}")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"[BLIP] Done. Saved to {output_file}")

# -------------------- Dataset Loader --------------------
class PromptDataset(Dataset):
    def __init__(self, img_dir, prompt_file, tokenizer, size=512):
        self.img_dir = Path(img_dir)
        self.tokenizer = tokenizer
        self.tfm_img = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.data = []
        for line in Path(prompt_file).read_text().splitlines():
            if not line: continue
            img, prompt = line.split("|", 1)
            p = self.img_dir / img
            if p.exists():
                self.data.append((p, prompt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p_img, prompt = self.data[idx]
        img = self.tfm_img(Image.open(p_img).convert("RGB"))
        ids = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids[0]
        return {"pixel_values": img, "input_ids": ids}

# -------------------- Evaluation Metrics --------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
lpips_model = LPIPS(net='alex').to(device).eval()
to_lpips_tensor = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@torch.inference_mode()
def clip_sim(img1, img2):
    inp = clip_processor(text=["a photo of a disaster road scene"], images=[img1, img2], return_tensors="pt").to(device)
    out = clip_model(**inp)
    return torch.cosine_similarity(out.image_embeds[0], out.image_embeds[1], dim=0).item()

@torch.inference_mode()
def lpips_dist(img1, img2):
    t1 = to_lpips_tensor(img1).unsqueeze(0).to(device)
    t2 = to_lpips_tensor(img2).unsqueeze(0).to(device)
    return lpips_model(t1, t2).item()

# -------------------- t-SNE Visualization --------------------
def run_tsne_visualization(train_img_dir, gen_img_dir, clip_model, clip_processor, save_path="tsne_result.png", sample_size=100):
    def extract_clip_embedding(img_path):
        image = Image.open(img_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs).squeeze().cpu().numpy()
        return emb

    features, labels = [], []
    train_images = sorted(Path(train_img_dir).glob("*.jpg"))[:sample_size]
    for img_path in train_images:
        features.append(extract_clip_embedding(img_path))
        labels.append("train")

    gen_images = sorted(Path(gen_img_dir).glob("*.jpg"))[:sample_size]
    for img_path in gen_images:
        features.append(extract_clip_embedding(img_path))
        labels.append("generated")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(np.array(features))

    plt.figure(figsize=(10, 7))
    colors = {"train": "blue", "generated": "red"}
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(tsne_result[idxs, 0], tsne_result[idxs, 1], c=colors[label], label=label, alpha=0.6)

    plt.legend()
    plt.title("t-SNE of CLIP Embeddings (Train vs Generated)")
    plt.savefig(os.path.join(gen_img_dir, save_path))
    print(f"[✓] t-SNE saved to {save_path}")

# -------------------- Main Function --------------------
def train_and_generate():
    generate_prompts_with_blip(TRAIN_IMG_DIR, PROMPT_FILE)

    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_PATH, subfolder="tokenizer")
    ds = PromptDataset(TRAIN_IMG_DIR, PROMPT_FILE, tokenizer, IMAGE_SIZE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL_PATH, torch_dtype=torch.float32).to(device)
    pipe.safety_checker = None

    unet, vae, text_encoder = pipe.unet, pipe.vae, pipe.text_encoder
    noise_sched = DDPMScheduler.from_pretrained(SD_MODEL_PATH, subfolder="scheduler")
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-5)

    accelerator = Accelerator()
    unet, vae, text_encoder, opt, dl = accelerator.prepare(unet, vae, text_encoder, opt, dl)

    best_ckpt = os.path.join(LORA_OUTPUT_DIR, "best_unet.pt")
    best_score = -float("inf")
    no_imp = 0
    start_epoch = 0

    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu")
        unet.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        best_score = ckpt["score"]
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, EPOCHS):
        unet.train()
        for batch in tqdm(dl, desc=f"Epoch {epoch}"):
            with accelerator.accumulate(unet):
                imgs = batch["pixel_values"].to(dtype=unet.dtype, device=unet.device)
                lat = vae.encode(imgs).latent_dist.sample() * 0.18215
                noise = torch.randn_like(lat)
                t = torch.randint(0, 1000, (lat.size(0),), device=lat.device).long()
                noisy = noise_sched.add_noise(lat, noise, t)
                enc_hid = text_encoder(batch["input_ids"])[0]
                pred = unet(noisy, t, encoder_hidden_states=enc_hid).sample
                loss = torch.nn.functional.mse_loss(pred, noise)
                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()

        accelerator.wait_for_everyone()

        unet.eval()
        pipe.unet = unet
        samp = random.sample(ds.data, min(10, len(ds)))

        clip_scores, lpips_scores = [], []
        for p_img, prompt in samp:
            ref = Image.open(p_img).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            gen = pipe(prompt, guidance_scale=GUIDANCE_SCALE, num_inference_steps=STEPS).images[0]
            c = clip_sim(gen, ref)
            l = lpips_dist(gen, ref)
            clip_scores.append(c)
            lpips_scores.append(l)

        avg_clip = np.mean(clip_scores)
        avg_lpips = np.mean(lpips_scores)
        avg_score = avg_clip - avg_lpips

        if avg_score > best_score:
            best_score, no_imp = avg_score, 0
            torch.save({"model": unet.state_dict(), "opt": opt.state_dict(), "score": best_score, "epoch": epoch}, best_ckpt)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    ckpt = torch.load(best_ckpt, map_location="cpu")
    unet.load_state_dict(ckpt["model"])
    pipe.unet = unet.eval().to(device)

    PROMPTS = [
        "an aerial view of a destroyed road and buildings",
        "an aerial view of a destroyed road and cars",
        "a view of a building and a road that has been destroyed",
        "an aerial view of houses and a road that have been destroyed",
        "an aerial view of houses and a road that were destroyed by hurricane florence",
    ]

    metrics = []
    saved, attempt = 0, 0
    while saved < NUM_GEN_IMAGES:
        attempt += 1
        prompt = random.choice(PROMPTS)
        ref_p, _ = random.choice(ds.data)
        ref = Image.open(ref_p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        seed = int(time.time() * 1000) % (2**32)
        generator = torch.Generator(device=device).manual_seed(seed)

        gen = pipe(prompt, guidance_scale=GUIDANCE_SCALE, num_inference_steps=STEPS, generator=generator).images[0]

        c, l = clip_sim(gen, ref), lpips_dist(gen, ref)
        if c > CLIP_TH and np.array(gen).mean() > 5:
            fname = f"gen_{saved:03d}.jpg"
            gen.save(Path(FINAL_GEN_DIR)/fname)
            metrics.append({"file": fname, "prompt": prompt, "clip": c, "lpips": l, "seed": seed})
            saved += 1

    pd.DataFrame(metrics).to_csv(Path(FINAL_GEN_DIR)/"metrics.csv", index=False)
    fid = calculate_fid_given_paths([TRAIN_IMG_DIR, FINAL_GEN_DIR], batch_size=32, device=device, dims=2048)
    Path(FINAL_GEN_DIR/"fid.txt").write_text(f"FID={fid:.2f}\n")
    run_tsne_visualization(TRAIN_IMG_DIR, FINAL_GEN_DIR, clip_model, clip_processor)

if __name__ == "__main__":
    train_and_generate()
