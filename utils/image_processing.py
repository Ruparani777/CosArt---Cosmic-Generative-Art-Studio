"""
CosArt - Image Processing Utilities (FIXED)
utils/image_processing.py
"""

import torch
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Union


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert PyTorch tensor to PIL Image"""
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    np_img = tensor.cpu().numpy()

    if np_img.shape[0] == 3:
        np_img = np.transpose(np_img, (1, 2, 0))

    np_img = (np_img * 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img, mode='RGB')

    return pil_img


def pil_to_tensor(pil_img: Image.Image, device: str = 'cpu') -> torch.Tensor:
    """Convert PIL Image to PyTorch tensor"""
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    np_img = np.array(pil_img).astype(np.float32) / 255.0
    np_img = np.transpose(np_img, (2, 0, 1))
    tensor = torch.from_numpy(np_img).to(device)
    tensor = tensor * 2 - 1

    return tensor


def pil_to_base64(pil_img: Image.Image, format: str = 'PNG') -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format, quality=95)
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    return f"data:image/{format.lower()};base64,{img_b64}"


def base64_to_pil(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    if b64_string.startswith('data:image'):
        b64_string = b64_string.split(',', 1)[1]

    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(io.BytesIO(img_bytes))

    return pil_img


def create_image_grid(
    images: List[Union[torch.Tensor, Image.Image]],
    grid_size: tuple = None
) -> Image.Image:
    """Create a grid of images"""
    pil_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            pil_images.append(tensor_to_pil(img))
        else:
            pil_images.append(img)

    if grid_size is None:
        n = len(pil_images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        grid_size = (rows, cols)

    rows, cols = grid_size
    img_width, img_height = pil_images[0].size
    grid_width = cols * img_width
    grid_height = rows * img_height
    grid = Image.new('RGB', (grid_width, grid_height), color=(0, 0, 0))

    for idx, img in enumerate(pil_images):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))

    return grid


# ============================================
# QUICK INSTALL SCRIPT
# ============================================

print("""
╔════════════════════════════════════════════════════════════╗
║          🌌 CosArt - Quick Fix Installation 🌌             ║
╚════════════════════════════════════════════════════════════╝

Run these commands in order:

1️⃣  Install Core Dependencies:
   pip install fastapi uvicorn[standard] pydantic pydantic-settings Pillow numpy

2️⃣  Install Additional Required:
   pip install python-multipart aiofiles websockets python-dateutil

3️⃣  Install PyTorch (CPU version - easier):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

4️⃣  Install ML Libraries:
   pip install scikit-learn scipy

5️⃣  Verify Installation:
   python -c "import fastapi, torch, PIL; print('✅ All packages installed!')"

6️⃣  Run CosArt:
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

═══════════════════════════════════════════════════════════════

🚀 FASTEST INSTALL (One Command):

pip install fastapi uvicorn[standard] pydantic pydantic-settings python-multipart Pillow numpy aiofiles websockets python-dateutil torch torchvision --index-url https://download.pytorch.org/whl/cpu scikit-learn scipy

═══════════════════════════════════════════════════════════════

⚠️  WINDOWS SPECIFIC ISSUES:

If you get errors on Windows, try:
1. Run PowerShell as Administrator
2. Use: pip install --user [package-name]
3. Or use conda: conda install -c pytorch pytorch torchvision

═══════════════════════════════════════════════════════════════
""")