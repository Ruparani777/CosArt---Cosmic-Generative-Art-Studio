"""
CosArt - Cosmic Generative Art Platform
Main FastAPI Application
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import torch
import numpy as np
from datetime import datetime
import io
from PIL import Image
import json

# Import custom modules (will be created)
from models.cosmic_stylegan import CosmicStyleGAN2
from inference.generator import ArtGenerator
from inference.universe_builder import UniverseNavigator
from cosmic.presets import PhysicsMapper
from cosmic.presets import CosmicPresets
from utils.image_processing import tensor_to_pil, create_image_grid
from config.settings import Settings

# Initialize FastAPI
app = FastAPI(
    title="CosArt API",
    description="Generate cosmic art through physics-inspired GANs",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
settings = Settings()
generator = None
universe_navigator = None
physics_mapper = PhysicsMapper()
cosmic_presets = CosmicPresets()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()


# Pydantic Models
class GenerationRequest(BaseModel):
    preset: Optional[str] = "nebula"
    seed: Optional[int] = None
    resolution: int = Field(512, ge=512, le=2048)
    physics_params: Optional[Dict[str, float]] = None
    batch_size: int = Field(1, ge=1, le=16)

class PhysicsParams(BaseModel):
    entropy: float = Field(0.5, ge=0.0, le=1.0)  # Chaos ↔ Order
    warp: float = Field(0.5, ge=0.0, le=1.0)  # Spacetime curvature
    luminosity: float = Field(0.7, ge=0.0, le=1.0)  # Energy density
    cosmic_flow: float = Field(0.5, ge=0.0, le=1.0)  # Expansion rate
    pattern_collapse: float = Field(0.3, ge=0.0, le=1.0)  # Symmetry breaking
    attraction: float = Field(0.5, ge=0.0, le=1.0)  # Gravitational force
    uncertainty: float = Field(0.2, ge=0.0, le=1.0)  # Quantum fluctuation
    spectral_shift: float = Field(0.5, ge=0.0, le=1.0)  # Redshift

class UniverseRequest(BaseModel):
    num_artworks: int = Field(50, ge=10, le=500)
    coherence: float = Field(0.8, ge=0.0, le=1.0)
    evolution_steps: int = Field(10, ge=1, le=100)
    base_preset: str = "cosmic_deep_space"

class InterpolationRequest(BaseModel):
    seed_start: int
    seed_end: int
    steps: int = Field(10, ge=2, le=50)
    physics_params: Optional[Dict[str, float]] = None


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global generator, universe_navigator
    
    print("🌌 Initializing CosArt...")
    print("✅ CosArt initialized successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🌙 Shutting down CosArt...")


# Health Check
@app.get("/")
async def root():
    return {
        "message": "🌌 Welcome to CosArt API",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "Cosmic GAN Generation",
            "Physics-Based Controls", 
            "Universe Mode Navigation",
            "Style Interpolation",
            "Custom Training"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "CosArt API is running",
        "version": "1.0.0"
    }


# Generation Endpoints
@app.post("/api/generate")
async def generate_art(request: GenerationRequest):
    """
    Generate cosmic artwork with physics-based controls
    """
    if generator is None:
        raise HTTPException(status_code=503, message="Generator not initialized")
    
    try:
        # Get preset configuration
        preset_config = cosmic_presets.get_preset(request.preset)
        
        # Merge with custom physics params
        physics_params = preset_config.get("physics_params", {})
        if request.physics_params:
            physics_params.update(request.physics_params)
        
        # Generate
        result = await generator.generate(
            seed=request.seed,
            resolution=request.resolution,
            physics_params=physics_params,
            batch_size=request.batch_size
        )
        
        return {
            "success": True,
            "generation_id": result["id"],
            "images": result["images"],  # Base64 encoded
            "metadata": {
                "seed": result["seed"],
                "preset": request.preset,
                "physics_params": physics_params,
                "resolution": request.resolution,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/universe")
async def generate_universe(request: UniverseRequest):
    """
    Generate a complete coherent universe (50-500 artworks)
    This is the KILLER FEATURE
    """
    if generator is None or universe_navigator is None:
        raise HTTPException(status_code=503, message="System not ready")
    
    try:
        # Generate universe with coherent latent DNA
        universe = await universe_navigator.generate_universe(
            num_artworks=request.num_artworks,
            coherence=request.coherence,
            evolution_steps=request.evolution_steps,
            base_preset=request.base_preset
        )
        
        return {
            "success": True,
            "universe_id": universe["id"],
            "num_artworks": len(universe["artworks"]),
            "preview_images": universe["previews"][:9],  # First 9 for preview
            "navigation_map": universe["3d_positions"],
            "metadata": {
                "coherence_score": universe["coherence_score"],
                "created_at": datetime.utcnow().isoformat(),
                "base_preset": request.base_preset
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interpolate")
async def cosmic_evolution(request: InterpolationRequest):
    """
    Create smooth transition between two artworks (Cosmic Evolution)
    """
    try:
        result = await generator.interpolate(
            seed_start=request.seed_start,
            seed_end=request.seed_end,
            steps=request.steps,
            physics_params=request.physics_params
        )
        
        return {
            "success": True,
            "frames": result["frames"],  # Base64 encoded images
            "animation_url": result.get("video_url"),  # Optional video
            "metadata": {
                "steps": request.steps,
                "duration": f"{request.steps * 0.5}s"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/universe/navigate")
async def navigate_universe(x: float, y: float, z: float, radius: float = 0.5):
    """
    Navigate 3D universe and find nearby artworks
    Universe Mode feature
    """
    if universe_navigator is None:
        raise HTTPException(status_code=503, message="Universe not initialized")
    
    try:
        nearby = universe_navigator.navigate(
            position=(x, y, z),
            radius=radius
        )
        
        return {
            "success": True,
            "position": {"x": x, "y": y, "z": z},
            "artworks_found": len(nearby),
            "previews": nearby[:12],  # Up to 12 previews
            "navigation_hints": universe_navigator.get_nearby_constellations((x, y, z))
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/presets")
async def get_cosmic_presets():
    """
    Get all available cosmic presets
    """
    return {
        "presets": cosmic_presets.list_all(),
        "categories": cosmic_presets.get_categories()
    }


@app.get("/api/presets/{preset_name}")
async def get_preset_details(preset_name: str):
    """
    Get detailed configuration for a specific preset
    """
    preset = cosmic_presets.get_preset(preset_name)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    
    return preset


# WebSocket for Real-time Generation Progress
@app.websocket("/ws/generation")
async def websocket_generation(websocket: WebSocket):
    """
    WebSocket endpoint for real-time generation progress
    """
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "generate":
                # Start generation with progress updates
                request = GenerationRequest(**data.get("params", {}))
                
                async for progress in generator.generate_streaming(
                    seed=request.seed,
                    resolution=request.resolution,
                    physics_params=request.physics_params
                ):
                    await websocket.send_json({
                        "type": "progress",
                        "progress": progress["percentage"],
                        "preview": progress.get("preview_image"),
                        "status": progress["status"]
                    })
                
                # Send final result
                await websocket.send_json({
                    "type": "complete",
                    "result": progress["final_result"]
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Style Mixing
@app.post("/api/mix")
async def gravitational_merge(
    seeds: List[int],
    weights: List[float],
    physics_params: Optional[Dict[str, float]] = None
):
    """
    Mix multiple styles (Gravitational Merging)
    """
    if len(seeds) != len(weights):
        raise HTTPException(status_code=400, detail="Seeds and weights must match")
    
    try:
        result = await generator.mix_styles(
            seeds=seeds,
            weights=weights,
            physics_params=physics_params
        )
        
        return {
            "success": True,
            "image": result["image"],
            "metadata": {
                "sources": seeds,
                "weights": weights
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Export endpoint
@app.post("/api/export/{generation_id}")
async def export_artwork(
    generation_id: str,
    format: str = "png",
    resolution: int = 2048,
    include_metadata: bool = True
):
    """
    Export artwork in various formats with optional metadata
    """
    try:
        # Retrieve generation
        artwork = await generator.get_generation(generation_id)
        
        if not artwork:
            raise HTTPException(status_code=404, detail="Generation not found")
        
        # Export with options
        exported = await generator.export(
            artwork=artwork,
            format=format,
            resolution=resolution,
            include_metadata=include_metadata
        )
        
        return StreamingResponse(
            io.BytesIO(exported["data"]),
            media_type=f"image/{format}",
            headers={
                "Content-Disposition": f"attachment; filename=cosart_{generation_id}.{format}"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Training endpoints
@app.post("/api/train/custom")
async def start_custom_training(
    name: str,
    dataset: UploadFile = File(...),
    base_model: str = "cosmic_base",
    epochs: int = 100
):
    """
    Start custom model training on user dataset
    """
    # Implementation for custom training
    # This would be a long-running task, typically handled via Celery
    return {
        "message": "Training started",
        "job_id": "train_" + name,
        "estimated_time": f"{epochs * 2} minutes"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )