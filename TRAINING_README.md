# CosArt Model Training Guide

## Current Status
- [x] Mock training completed
- [ ] Real training pending (needs dataset)

## To Train Real Models:

### 1. Prepare Dataset
```
# Create directory
mkdir -p data/cosmic_images

# Add cosmic images (nebulae, galaxies, black holes, etc.)
# Minimum: 10,000 images
# Recommended: 50,000+ images
# Sources: NASA, ESA, Hubble images, synthetic generation
```

### 2. Hardware Requirements
- GPU: NVIDIA with 8GB+ VRAM (RTX 3060 or better)
- RAM: 16GB+
- Storage: 100GB+ for dataset and checkpoints
- Time: 1-2 weeks for initial training

### 3. Run Training
```
# Install GPU version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Start training
python train.py
```

### 4. Monitor Progress
- Check checkpoints/ directory for saved models
- Loss should decrease over time
- Generate sample images to check quality

## Pre-trained Models
For immediate results, download pre-trained cosmic StyleGAN weights from:
- Model repositories (when available)
- Research institutions
- Community shared models

## Expected Results
After training:
- Beautiful cosmic art generation
- Physics controls produce meaningful variations
- Universe Mode shows real latent space structure
- High-resolution outputs (up to 4096x4096)
