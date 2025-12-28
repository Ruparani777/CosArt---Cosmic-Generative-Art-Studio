""""
CosArt - Art Generation Engine
inference/generator.py

Handles all generation logic, caching, and image processing
"""
import torch
import numpy as np
from typing import Dict, Optional, List, AsyncIterator
from PIL import Image
import io
import base64
import uuid
from datetime import datetime
import asyncio

from models.cosmic_stylegan import CosmicStyleGAN2, create_cosmic_stylegan
from cosmic.presets import PhysicsMapper
from cosmic.presets import CosmicPresets
from utils.image_processing import tensor_to_pil, pil_to_base64, create_image_grid


class ArtGenerator:
    """
    Main art generation engine for CosArt
    Manages models, caching, and generation workflows
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.generation_cache = {}
        self.physics_mapper = PhysicsMapper()
        self.presets = CosmicPresets()
        
        print(f"🎨 Initializing ArtGenerator on {self.device}")
    
    async def load_models(self):
        """
        Load pre-trained models
        In production, these would be downloaded from cloud storage
        """
        print("📦 Loading Cosmic GAN models...")
        
        # Load main model
        self.models['cosmic_base'] = create_cosmic_stylegan(
            resolution=1024,
            pretrained=False,  # Set to True when weights available
            device=str(self.device)
        )
        
        # Load specialized models (in production)
        # self.models['cosmic_nebula'] = ...
        # self.models['cosmic_galaxy'] = ...
        
        print(f"✅ Loaded {len(self.models)} models")
    
    async def generate(
        self,
        seed: Optional[int] = None,
        resolution: int = 512,
        physics_params: Optional[Dict[str, float]] = None,
        batch_size: int = 1,
        model_name: str = 'cosmic_base'
    ) -> Dict:
        """
        Generate cosmic artwork
        
        Args:
            seed: Random seed for reproducibility
            resolution: Output resolution (512, 1024, 2048)
            physics_params: Dict of cosmic control parameters
            batch_size: Number of images to generate
            model_name: Which model to use
        
        Returns:
            Dict with generation results
        """
        if seed is None:
            seed = np.random.randint(0, 1000000)
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Get model
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Generate latent codes
        z = torch.randn(batch_size, 512, device=self.device)
        
        # Apply physics parameters
        if physics_params is None:
            physics_params = {}
        
        # Generate images
        with torch.no_grad():
            images_tensor = model(z, physics_params)
        
        # Convert to PIL images
        images_pil = [tensor_to_pil(img) for img in images_tensor]
        
        # Resize if needed
        if resolution != images_pil[0].size[0]:
            images_pil = [img.resize((resolution, resolution), Image.LANCZOS) 
                         for img in images_pil]
        
        # Convert to base64
        images_b64 = [pil_to_base64(img) for img in images_pil]
        
        # Create result
        generation_id = str(uuid.uuid4())
        result = {
            'id': generation_id,
            'images': images_b64,
            'seed': seed,
            'resolution': resolution,
            'physics_params': physics_params,
            'batch_size': batch_size,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Cache result
        self.generation_cache[generation_id] = result
        
        return result
    
    async def generate_streaming(
        self,
        seed: Optional[int] = None,
        resolution: int = 512,
        physics_params: Optional[Dict[str, float]] = None
    ) -> AsyncIterator[Dict]:
        """
        Generate with progress updates (for WebSocket)
        """
        # Simulate progressive generation
        for progress in range(0, 101, 10):
            await asyncio.sleep(0.3)  # Simulate processing time
            
            yield {
                'percentage': progress,
                'status': 'generating' if progress < 100 else 'complete',
                'preview_image': None,  # Could add low-res preview
            }
        
        # Final generation
        result = await self.generate(seed, resolution, physics_params)
        
        yield {
            'percentage': 100,
            'status': 'complete',
            'final_result': result
        }
    
    async def interpolate(
        self,
        seed_start: int,
        seed_end: int,
        steps: int = 10,
        physics_params: Optional[Dict[str, float]] = None,
        resolution: int = 512
    ) -> Dict:
        """
        Cosmic Evolution: Interpolate between two seeds
        """
        model = self.models['cosmic_base']
        
        # Generate start and end latents
        torch.manual_seed(seed_start)
        z1 = torch.randn(1, 512, device=self.device)
        
        torch.manual_seed(seed_end)
        z2 = torch.randn(1, 512, device=self.device)
        
        # Interpolate
        frames = []
        alphas = torch.linspace(0, 1, steps)
        
        with torch.no_grad():
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                img_tensor = model(z_interp, physics_params)
                
                img_pil = tensor_to_pil(img_tensor[0])
                if resolution != img_pil.size[0]:
                    img_pil = img_pil.resize((resolution, resolution), Image.LANCZOS)
                
                frames.append(pil_to_base64(img_pil))
        
        return {
            'frames': frames,
            'num_frames': len(frames),
            'seed_start': seed_start,
            'seed_end': seed_end
        }
    
    async def mix_styles(
        self,
        seeds: List[int],
        weights: List[float],
        physics_params: Optional[Dict[str, float]] = None,
        resolution: int = 512
    ) -> Dict:
        """
        Gravitational Merging: Mix multiple styles
        """
        model = self.models['cosmic_base']
        
        # Generate latents for each seed
        latents = []
        for seed in seeds:
            torch.manual_seed(seed)
            z = torch.randn(1, 512, device=self.device)
            latents.append(z)
        
        # Mix using model's built-in method
        with torch.no_grad():
            img_tensor = model.mix_styles(latents, weights, physics_params)
        
        img_pil = tensor_to_pil(img_tensor[0])
        if resolution != img_pil.size[0]:
            img_pil = img_pil.resize((resolution, resolution), Image.LANCZOS)
        
        return {
            'image': pil_to_base64(img_pil),
            'seeds': seeds,
            'weights': weights
        }
    
    async def get_generation(self, generation_id: str) -> Optional[Dict]:
        """
        Retrieve a cached generation
        """
        return self.generation_cache.get(generation_id)
    
    async def export(
        self,
        artwork: Dict,
        format: str = 'png',
        resolution: int = 2048,
        include_metadata: bool = True
    ) -> Dict:
        """
        Export artwork in high resolution with metadata
        """
        # Decode base64 image
        img_data = base64.b64decode(artwork['images'][0])
        img = Image.open(io.BytesIO(img_data))
        
        # Upscale if needed
        if resolution > img.size[0]:
            img = img.resize((resolution, resolution), Image.LANCZOS)
        
        # Add metadata if requested
        if include_metadata:
            from PIL import PngImagePlugin
            meta = PngImagePlugin.PngInfo()
            meta.add_text("CosArt:Seed", str(artwork['seed']))
            meta.add_text("CosArt:Physics", str(artwork['physics_params']))
            meta.add_text("CosArt:Timestamp", artwork['timestamp'])
        else:
            meta = None
        
        # Save to bytes
        output = io.BytesIO()
        img.save(output, format=format.upper(), pnginfo=meta)
        output.seek(0)
        
        return {
            'data': output.getvalue(),
            'format': format,
            'resolution': resolution
        }


class ModelCache:
    """
    Manages model loading and memory
    """
    def __init__(self, max_models: int = 3):
        self.cache = {}
        self.max_models = max_models
        self.access_count = {}
    
    def get(self, model_name: str):
        if model_name in self.cache:
            self.access_count[model_name] += 1
            return self.cache[model_name]
        return None
    
    def put(self, model_name: str, model):
        if len(self.cache) >= self.max_models:
            # Remove least recently used
            lru = min(self.access_count, key=self.access_count.get)
            del self.cache[lru]
            del self.access_count[lru]
        
        self.cache[model_name] = model
        self.access_count[model_name] = 1


# Batch generation for efficiency
class BatchGenerator:
    """
    Handles batch generation with queue management
    """
    def __init__(self, generator: ArtGenerator, max_batch_size: int = 16):
        self.generator = generator
        self.max_batch_size = max_batch_size
        self.queue = []
    
    async def add_to_queue(self, request: Dict):
        """Add generation request to queue"""
        self.queue.append(request)
        
        if len(self.queue) >= self.max_batch_size:
            await self.process_batch()
    
    async def process_batch(self):
        """Process accumulated requests in batch"""
        if not self.queue:
            return
        
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        
        # Process batch efficiently
        # Implementation depends on specific batching strategy
        pass


if __name__ == "__main__":
    # Test generator
    async def test():
        print("🧪 Testing ArtGenerator...")
        
        gen = ArtGenerator(device="cpu")  # Use CPU for testing
        await gen.load_models()
        
        # Test generation
        result = await gen.generate(
            seed=12345,
            resolution=512,
            physics_params={
                'entropy': 0.7,
                'warp': 0.5,
                'luminosity': 0.9
            }
        )
        
        print(f"✅ Generated {len(result['images'])} image(s)")
        print(f"✅ Generation ID: {result['id']}")
    
    import asyncio
    asyncio.run(test())