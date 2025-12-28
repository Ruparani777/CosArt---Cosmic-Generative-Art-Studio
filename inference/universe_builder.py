"""
CosArt - Universe Navigator
inference/universe_builder.py

The KILLER FEATURE: 3D navigation through latent space
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import KDTree, distance
import uuid
from datetime import datetime


class UniverseNavigator:
    """
    Navigate through GAN latent space as a 3D universe
    Each point in space represents a unique artwork
    """
    
    def __init__(self, generator, n_samples: int = 10000):
        """
        Args:
            generator: ArtGenerator instance
            n_samples: Number of points to pre-compute for universe
        """
        self.generator = generator
        self.n_samples = n_samples
        self.universe_map = None
        self.constellations = {}
        
        print(f"🗺️  Building universe with {n_samples} artworks...")
        self._build_universe()
    
    def _build_universe(self):
        """
        Pre-compute 3D map of latent space using dimensionality reduction
        """
        model = self.generator.models['cosmic_base']
        device = self.generator.device
        
        # Sample random latent codes
        print("🎲 Sampling latent space...")
        z_samples = torch.randn(self.n_samples, 512, device=device)
        
        # Map to w space
        print("🌊 Mapping to intermediate space...")
        with torch.no_grad():
            w_samples = model.mapping_network(z_samples)
        
        w_np = w_samples.cpu().numpy()
        
        # Reduce to 3D using PCA (fast) + TSNE (quality)
        print("📐 Reducing to 3D coordinates...")
        
        # First PCA to 50D for speed
        pca_50 = PCA(n_components=50)
        w_reduced = pca_50.fit_transform(w_np)
        
        # Then PCA to 3D for final positions
        pca_3 = PCA(n_components=3)
        positions_3d = pca_3.fit_transform(w_reduced)
        
        # Normalize to [-1, 1] cube
        positions_3d = (positions_3d - positions_3d.min(axis=0)) / \
                       (positions_3d.max(axis=0) - positions_3d.min(axis=0))
        positions_3d = positions_3d * 2 - 1
        
        # Build KD-tree for fast spatial queries
        print("🌳 Building spatial index...")
        tree = KDTree(positions_3d)
        
        # Identify constellations (clusters)
        print("✨ Discovering constellations...")
        self._find_constellations(positions_3d, w_samples)
        
        self.universe_map = {
            'positions': positions_3d,
            'latents_z': z_samples.cpu(),
            'latents_w': w_samples.cpu(),
            'tree': tree,
            'pca_50': pca_50,
            'pca_3': pca_3
        }
        
        print("✅ Universe map built successfully!")
    
    def _find_constellations(
        self,
        positions: np.ndarray,
        latents: torch.Tensor,
        n_clusters: int = 12
    ):
        """
        Identify constellation clusters in the universe
        Named after cosmic phenomena
        """
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(positions)
        
        constellation_names = [
            "Nebula Prime", "Dark Matter Void", "Supernova Cluster",
            "Quasar Region", "Black Hole Basin", "Cosmic Web Node",
            "Stellar Nursery", "Galaxy Supercluster", "Antimatter Anomaly",
            "Wormhole Nexus", "Pulsar Field", "Event Horizon"
        ]
        
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_positions = positions[cluster_mask]
            cluster_center = cluster_positions.mean(axis=0)
            
            self.constellations[constellation_names[i]] = {
                'center': cluster_center.tolist(),
                'radius': np.std(cluster_positions, axis=0).mean(),
                'num_artworks': cluster_mask.sum(),
                'members': np.where(cluster_mask)[0].tolist()
            }
    
    def navigate(
        self,
        position: Tuple[float, float, float],
        radius: float = 0.5
    ) -> List[Dict]:
        """
        Find artworks near a 3D position
        
        Args:
            position: (x, y, z) coordinates in [-1, 1] cube
            radius: Search radius
        
        Returns:
            List of nearby artworks with preview data
        """
        if self.universe_map is None:
            raise RuntimeError("Universe not built yet")
        
        # Find nearby points using KD-tree
        indices = self.universe_map['tree'].query_ball_point(position, radius)
        
        # Limit to reasonable number
        indices = indices[:50] if len(indices) > 50 else indices
        
        # Generate previews for nearby artworks
        nearby = []
        for idx in indices:
            z = self.universe_map['latents_z'][idx].unsqueeze(0).to(self.generator.device)
            
            # Quick low-res preview
            with torch.no_grad():
                model = self.generator.models['cosmic_base']
                img = model(z)
            
            # Convert to base64 preview (low res for speed)
            from utils.image_processing import tensor_to_pil, pil_to_base64
            img_pil = tensor_to_pil(img[0]).resize((128, 128))
            img_b64 = pil_to_base64(img_pil)
            
            nearby.append({
                'index': int(idx),
                'position': self.universe_map['positions'][idx].tolist(),
                'distance': float(distance.euclidean(position, self.universe_map['positions'][idx])),
                'preview': img_b64
            })
        
        # Sort by distance
        nearby.sort(key=lambda x: x['distance'])
        
        return nearby
    
    def get_nearby_constellations(
        self,
        position: Tuple[float, float, float]
    ) -> List[Dict]:
        """
        Find which constellations are nearby
        """
        nearby_constellations = []
        
        for name, info in self.constellations.items():
            dist = distance.euclidean(position, info['center'])
            
            if dist < info['radius'] * 2:  # Within 2x radius
                nearby_constellations.append({
                    'name': name,
                    'distance': float(dist),
                    'center': info['center'],
                    'num_artworks': info['num_artworks']
                })
        
        nearby_constellations.sort(key=lambda x: x['distance'])
        return nearby_constellations
    
    async def generate_universe(
        self,
        num_artworks: int = 50,
        coherence: float = 0.8,
        evolution_steps: int = 10,
        base_preset: str = "cosmic_deep_space"
    ) -> Dict:
        """
        KILLER FEATURE: Generate a complete coherent universe
        
        Args:
            num_artworks: Number of artworks to generate (10-500)
            coherence: How similar artworks should be (0=random, 1=identical)
            evolution_steps: Steps of evolution from seed
            base_preset: Cosmic preset to base universe on
        
        Returns:
            Complete universe with all artworks and metadata
        """
        print(f"🌌 Generating universe with {num_artworks} artworks...")
        
        # Generate base "DNA" for this universe
        universe_seed = np.random.randint(0, 1000000)
        torch.manual_seed(universe_seed)
        
        # Create coherent latent space
        base_z = torch.randn(1, 512, device=self.generator.device)
        
        # Generate variations with controlled coherence
        artworks = []
        positions_3d = []
        
        for i in range(num_artworks):
            # Add controlled noise based on coherence
            noise_scale = (1 - coherence) * 0.5
            variation_z = base_z + torch.randn_like(base_z) * noise_scale
            
            # Evolve through steps
            if evolution_steps > 1:
                evolution_factor = i / num_artworks
                drift = torch.randn_like(base_z) * evolution_factor * 0.3
                variation_z = variation_z + drift
            
            # Generate artwork
            with torch.no_grad():
                model = self.generator.models['cosmic_base']
                img = model(variation_z)
            
            # Convert and store
            from utils.image_processing import tensor_to_pil, pil_to_base64
            img_pil = tensor_to_pil(img[0])
            img_b64 = pil_to_base64(img_pil)
            
            # Compute 3D position for this artwork
            w = model.mapping_network(variation_z)
            w_np = w.cpu().numpy().reshape(1, -1)
            w_reduced = self.universe_map['pca_50'].transform(w_np)
            pos_3d = self.universe_map['pca_3'].transform(w_reduced)[0]
            
            artworks.append({
                'id': f"{universe_seed}_{i}",
                'image': img_b64,
                'position': pos_3d.tolist(),
                'index': i
            })
            
            positions_3d.append(pos_3d)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_artworks}...")
        
        # Calculate coherence score
        positions_np = np.array(positions_3d)
        distances = distance.pdist(positions_np)
        coherence_score = 1.0 - (distances.mean() / distances.max())
        
        universe_id = str(uuid.uuid4())
        
        return {
            'id': universe_id,
            'universe_seed': universe_seed,
            'artworks': artworks,
            'previews': [a['image'] for a in artworks],
            '3d_positions': positions_3d,
            'coherence_score': float(coherence_score),
            'num_artworks': num_artworks,
            'evolution_steps': evolution_steps,
            'created_at': datetime.utcnow().isoformat()
        }
    
    def create_wormhole(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
        steps: int = 20
    ) -> List[Tuple[float, float, float]]:
        """
        Create a path between two points (wormhole travel)
        """
        # Interpolate with slight curve (not straight line)
        t = np.linspace(0, 1, steps)
        
        # Add bezier-like curve
        mid = tuple((np.array(pos1) + np.array(pos2)) / 2)
        mid_offset = np.random.randn(3) * 0.3  # Random midpoint offset
        mid = tuple(np.array(mid) + mid_offset)
        
        path = []
        for ti in t:
            # Quadratic bezier interpolation
            p = (1 - ti)**2 * np.array(pos1) + \
                2 * (1 - ti) * ti * np.array(mid) + \
                ti**2 * np.array(pos2)
            path.append(tuple(p))
        
        return path
    
    def quantum_drift(
        self,
        start_position: Tuple[float, float, float],
        duration: int = 100,
        drift_speed: float = 0.01
    ) -> List[Tuple[float, float, float]]:
        """
        Random walk through latent space (Quantum Drift mode)
        """
        path = [start_position]
        current = np.array(start_position)
        
        for _ in range(duration):
            # Random direction with momentum
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            # Move
            current = current + direction * drift_speed
            
            # Keep in bounds [-1, 1]
            current = np.clip(current, -1, 1)
            
            path.append(tuple(current))
        
        return path
    
    def get_universe_stats(self) -> Dict:
        """
        Get statistics about the mapped universe
        """
        return {
            'total_artworks': self.n_samples,
            'num_constellations': len(self.constellations),
            'constellations': {
                name: {
                    'center': info['center'],
                    'size': info['num_artworks']
                }
                for name, info in self.constellations.items()
            },
            'volume': float(np.prod(self.universe_map['positions'].max(axis=0) - 
                                   self.universe_map['positions'].min(axis=0)))
        }


if __name__ == "__main__":
    print("🧪 Universe Navigator requires ArtGenerator instance")
    print("✅ Module loaded successfully")