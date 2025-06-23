import os
import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import rasterio
import yaml

def save_prompt_yaml(prompt: str, dem_path: str, save_dir: str, file_name="log.yaml"):
    os.makedirs(save_dir, exist_ok=True)
    log_data = {
        "prompt": prompt,
        "dem": dem_path,
    }
    log_path = os.path.join(save_dir, file_name)
    with open(log_path, "a") as f:
        yaml.dump([log_data], f)

def load_dem_tif(path: str, size=(512, 512)) -> torch.Tensor:
    """Load .tif DEM image using rasterio, normalize and convert to tensor."""
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
    dem = np.nan_to_num(dem)  # NaN 제거
    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
    dem = Image.fromarray((dem * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    dem = transforms.ToTensor()(dem).unsqueeze(0)  # [1,1,H,W]
    return dem

def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_mask_image(mask_path: str, size=(512, 512)) -> torch.Tensor:
    """Load a binary mask image and convert to tensor."""
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(size, Image.BILINEAR)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return mask

def load_image(image_path: str, size=(512, 512)) -> torch.Tensor:
    """Load an RGB image and convert to tensor."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(size, Image.BILINEAR)
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)  # [1,3,H,W]

def load_dem(dem_path: str, size=(512, 512)) -> torch.Tensor:
    """Load DEM data as grayscale image (can be extended to .tif parsing)."""
    dem = Image.open(dem_path).convert("L")
    dem = dem.resize(size, Image.BILINEAR)
    dem = np.array(dem).astype(np.float32) / 255.0
    dem = torch.from_numpy(dem).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return dem

def save_image(image: Image.Image, save_dir: str, name: str):
    """Save generated image to the target directory."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    image.save(path)

def save_prompt_log(prompt: str, save_dir: str, file_name: str = "prompts.txt"):
    """Log prompts used for generation."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, file_name), 'a') as f:
        f.write(prompt + "\n")

def save_tensor_as_image(tensor: torch.Tensor, path: str):
    """Convert 1x1xHxW tensor to image and save."""
    arr = tensor.squeeze().cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def seed_everything(seed: int = 42):
    """Fix all random seeds for reproducibility across modules."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def check_size_match(*tensors):
    sizes = [t.shape[-2:] for t in tensors]
    assert all(s == sizes[0] for s in sizes), f"Input size mismatch: {sizes}"

def load_dem_rgb(dem_path: str, size=(512, 512)) -> Image.Image:
    dem = Image.open(dem_path).convert("L").resize(size, Image.BILINEAR)
    dem_rgb = Image.merge("RGB", (dem, dem, dem))  # ✅ 3채널
    return dem_rgb.convert("RGB")

    
