"""
CosArt - Cosmic Features Module
cosmic/presets.py & cosmic/physics_mapper.py

Defines all cosmic presets and physics-to-latent mappings
"""
import numpy as np
from typing import Dict, List
import torch


class CosmicPresets:
    """
    Predefined cosmic presets based on astrophysical phenomena
    """
    
    def __init__(self):
        self.presets = {
            # NEBULAE
            'nebula': {
                'name': '🌌 Nebula',
                'description': 'Colorful star-forming regions with flowing gas',
                'physics_params': {
                    'entropy': 0.6,
                    'warp': 0.4,
                    'luminosity': 0.8,
                    'cosmic_flow': 0.7,
                    'pattern_collapse': 0.3,
                    'attraction': 0.5,
                    'uncertainty': 0.4,
                    'spectral_shift': 0.6
                },
                'color_palette': 'pink_purple_blue'
            },
            
            # BLACK HOLES
            'black_hole': {
                'name': '🕳️ Black Hole',
                'description': 'Event horizon with accretion disk',
                'physics_params': {
                    'entropy': 0.3,
                    'warp': 0.95,  # Maximum spacetime curvature
                    'luminosity': 0.3,
                    'cosmic_flow': 0.2,
                    'pattern_collapse': 0.8,
                    'attraction': 0.95,  # Maximum gravitational pull
                    'uncertainty': 0.1,
                    'spectral_shift': 0.9
                },
                'color_palette': 'black_orange_white'
            },
            
            # GALAXIES
            'galaxy': {
                'name': '🌀 Galaxy',
                'description': 'Spiral structure with billions of stars',
                'physics_params': {
                    'entropy': 0.5,
                    'warp': 0.3,
                    'luminosity': 0.7,
                    'cosmic_flow': 0.4,
                    'pattern_collapse': 0.2,
                    'attraction': 0.6,
                    'uncertainty': 0.3,
                    'spectral_shift': 0.5
                },
                'color_palette': 'blue_white_yellow'
            },
            
            # STAR FIELDS
            'star_field': {
                'name': '✨ Star Field',
                'description': 'Dense stellar populations',
                'physics_params': {
                    'entropy': 0.8,
                    'warp': 0.2,
                    'luminosity': 0.9,
                    'cosmic_flow': 0.3,
                    'pattern_collapse': 0.1,
                    'attraction': 0.4,
                    'uncertainty': 0.6,
                    'spectral_shift': 0.4
                },
                'color_palette': 'blue_white_multi'
            },
            
            # DARK MATTER
            'dark_matter': {
                'name': '👻 Dark Matter',
                'description': 'Invisible cosmic scaffolding',
                'physics_params': {
                    'entropy': 0.4,
                    'warp': 0.6,
                    'luminosity': 0.2,
                    'cosmic_flow': 0.5,
                    'pattern_collapse': 0.6,
                    'attraction': 0.8,
                    'uncertainty': 0.7,
                    'spectral_shift': 0.3
                },
                'color_palette': 'dark_purple_black'
            },
            
            # COSMIC WEB
            'cosmic_web': {
                'name': '🕸️ Cosmic Web',
                'description': 'Large-scale structure of the universe',
                'physics_params': {
                    'entropy': 0.5,
                    'warp': 0.4,
                    'luminosity': 0.5,
                    'cosmic_flow': 0.6,
                    'pattern_collapse': 0.4,
                    'attraction': 0.7,
                    'uncertainty': 0.5,
                    'spectral_shift': 0.5
                },
                'color_palette': 'blue_purple_cyan'
            },
            
            # SUPERNOVA
            'supernova': {
                'name': '💥 Supernova',
                'description': 'Stellar explosion remnant',
                'physics_params': {
                    'entropy': 0.9,
                    'warp': 0.7,
                    'luminosity': 1.0,
                    'cosmic_flow': 0.9,
                    'pattern_collapse': 0.7,
                    'attraction': 0.3,
                    'uncertainty': 0.8,
                    'spectral_shift': 0.8
                },
                'color_palette': 'red_orange_white'
            },
            
            # COSMIC MICROWAVE BACKGROUND
            'cmb': {
                'name': '📡 CMB Pattern',
                'description': 'Cosmic microwave background fluctuations',
                'physics_params': {
                    'entropy': 0.2,
                    'warp': 0.1,
                    'luminosity': 0.4,
                    'cosmic_flow': 0.1,
                    'pattern_collapse': 0.1,
                    'attraction': 0.5,
                    'uncertainty': 0.9,
                    'spectral_shift': 0.2
                },
                'color_palette': 'red_yellow_blue'
            },
            
            # QUASAR
            'quasar': {
                'name': '🌟 Quasar',
                'description': 'Extremely luminous active galactic nucleus',
                'physics_params': {
                    'entropy': 0.7,
                    'warp': 0.8,
                    'luminosity': 1.0,
                    'cosmic_flow': 0.8,
                    'pattern_collapse': 0.6,
                    'attraction': 0.9,
                    'uncertainty': 0.5,
                    'spectral_shift': 1.0
                },
                'color_palette': 'blue_white_intense'
            },
            
            # EXOPLANET
            'exoplanet': {
                'name': '🪐 Exoplanet',
                'description': 'Alien world atmosphere',
                'physics_params': {
                    'entropy': 0.6,
                    'warp': 0.3,
                    'luminosity': 0.6,
                    'cosmic_flow': 0.4,
                    'pattern_collapse': 0.5,
                    'attraction': 0.5,
                    'uncertainty': 0.4,
                    'spectral_shift': 0.7
                },
                'color_palette': 'orange_brown_blue'
            }
        }
    
    def get_preset(self, name: str) -> Dict:
        """Get preset configuration"""
        return self.presets.get(name, self.presets['nebula'])
    
    def list_all(self) -> List[Dict]:
        """List all presets with metadata"""
        return [
            {
                'id': key,
                'name': preset['name'],
                'description': preset['description']
            }
            for key, preset in self.presets.items()
        ]
    
    def get_categories(self) -> Dict:
        """Group presets by category"""
        return {
            'Stellar Phenomena': ['nebula', 'star_field', 'supernova'],
            'Galactic': ['galaxy', 'black_hole', 'quasar'],
            'Cosmic Scale': ['cosmic_web', 'dark_matter', 'cmb'],
            'Planetary': ['exoplanet']
        }


class PhysicsMapper:
    """
    Maps physics concepts to GAN latent space modifications
    This is where the magic of physics-inspired art happens
    """
    
    def __init__(self):
        # Define which latent channels control which aspects
        self.parameter_mappings = {
            'entropy': {
                'channels': list(range(0, 64)),  # First 64 channels
                'transform': self._entropy_transform,
                'description': 'Chaos ↔ Order: Controls visual randomness vs structure'
            },
            'warp': {
                'channels': list(range(128, 192)),
                'transform': self._warp_transform,
                'description': 'Spacetime curvature: Bending and distortion'
            },
            'luminosity': {
                'channels': list(range(256, 320)),
                'transform': self._luminosity_transform,
                'description': 'Energy density: Brightness and glow'
            },
            'cosmic_flow': {
                'channels': list(range(320, 384)),
                'transform': self._flow_transform,
                'description': 'Expansion rate: Speed of change'
            },
            'pattern_collapse': {
                'channels': list(range(384, 448)),
                'transform': self._collapse_transform,
                'description': 'Symmetry breaking: Order to complexity'
            },
            'attraction': {
                'channels': list(range(64, 128)),
                'transform': self._attraction_transform,
                'description': 'Gravitational force: Element clustering'
            },
            'uncertainty': {
                'channels': list(range(448, 480)),
                'transform': self._uncertainty_transform,
                'description': 'Quantum fluctuation: Micro-variations'
            },
            'spectral_shift': {
                'channels': list(range(192, 256)),
                'transform': self._spectral_transform,
                'description': 'Redshift: Color palette movement'
            }
        }
    
    def _entropy_transform(self, value: float) -> float:
        """High entropy = more chaos, Low entropy = more order"""
        # Exponential transform: small changes at low values, large at high
        return np.power(value, 2) * 2 - 1
    
    def _warp_transform(self, value: float) -> float:
        """Spacetime curvature increases non-linearly"""
        return np.tanh(value * 3) * 2
    
    def _luminosity_transform(self, value: float) -> float:
        """Energy density follows Stefan-Boltzmann law (T^4)"""
        return np.power(value, 0.25) * 2 - 1
    
    def _flow_transform(self, value: float) -> float:
        """Cosmic expansion (Hubble flow)"""
        return value * 2 - 1
    
    def _collapse_transform(self, value: float) -> float:
        """Symmetry breaking (phase transition)"""
        # Sigmoid for sharp transition
        return 2 / (1 + np.exp(-10 * (value - 0.5))) - 1
    
    def _attraction_transform(self, value: float) -> float:
        """Gravitational attraction (inverse square-ish)"""
        return np.power(value, 3) * 2 - 1
    
    def _uncertainty_transform(self, value: float) -> float:
        """Quantum uncertainty (Heisenberg principle)"""
        # Add randomness scaled by uncertainty
        base = value * 2 - 1
        noise = np.random.randn() * value * 0.3
        return np.clip(base + noise, -1, 1)
    
    def _spectral_transform(self, value: float) -> float:
        """Doppler shift / Redshift"""
        # Maps to color temperature
        return np.sin(value * np.pi) * 2 - 1
    
    def map_to_latent(self, physics_params: Dict[str, float]) -> np.ndarray:
        """
        Convert physics parameters to latent space modification vector
        
        Args:
            physics_params: Dict with parameter names and values (0-1)
        
        Returns:
            Latent modification vector (512-dim)
        """
        latent_mod = np.zeros(512)
        
        for param_name, value in physics_params.items():
            if param_name not in self.parameter_mappings:
                continue
            
            mapping = self.parameter_mappings[param_name]
            channels = mapping['channels']
            transform = mapping['transform']
            
            # Apply transform
            transformed_value = transform(value)
            
            # Modify specific channels
            latent_mod[channels] += transformed_value
        
        # Normalize to prevent explosion
        latent_mod = latent_mod / (len(physics_params) + 1)
        
        return latent_mod
    
    def get_parameter_info(self, param_name: str) -> Dict:
        """Get information about a physics parameter"""
        if param_name in self.parameter_mappings:
            mapping = self.parameter_mappings[param_name]
            return {
                'name': param_name,
                'description': mapping['description'],
                'channels': len(mapping['channels']),
                'range': [0.0, 1.0]
            }
        return None
    
    def get_all_parameters(self) -> List[Dict]:
        """Get info for all physics parameters"""
        return [self.get_parameter_info(name) for name in self.parameter_mappings.keys()]


# Cosmic color palettes based on real astronomical data
class CosmicColorPalettes:
    """
    Color palettes derived from actual cosmic phenomena
    """
    
    PALETTES = {
        'pink_purple_blue': [
            '#FF1493',  # Deep Pink (H-alpha)
            '#8B00FF',  # Purple
            '#4169E1',  # Royal Blue
            '#00CED1',  # Dark Turquoise
        ],
        'black_orange_white': [
            '#000000',  # Black (event horizon)
            '#FF4500',  # Orange Red (accretion disk)
            '#FFD700',  # Gold
            '#FFFFFF',  # White (jets)
        ],
        'blue_white_yellow': [
            '#000080',  # Navy (dark space)
            '#4682B4',  # Steel Blue (arms)
            '#F0E68C',  # Khaki (old stars)
            '#FFFFFF',  # White (young stars)
        ],
        'dark_purple_black': [
            '#0A0A0A',  # Almost black
            '#1A0033',  # Very dark purple
            '#330066',  # Dark purple
            '#4B0082',  # Indigo
        ],
        'red_orange_white': [
            '#8B0000',  # Dark Red
            '#FF4500',  # Orange Red
            '#FFA500',  # Orange
            '#FFFFFF',  # White
        ]
    }
    
    @staticmethod
    def get_palette(name: str) -> List[str]:
        """Get color palette by name"""
        return CosmicColorPalettes.PALETTES.get(name, 
                   CosmicColorPalettes.PALETTES['pink_purple_blue'])


if __name__ == "__main__":
    # Test presets
    print("🧪 Testing Cosmic Presets...")
    
    presets = CosmicPresets()
    print(f"✅ Loaded {len(presets.presets)} presets")
    
    nebula = presets.get_preset('nebula')
    print(f"✅ Nebula preset: {nebula['name']}")
    
    # Test physics mapper
    print("\n🧪 Testing Physics Mapper...")
    
    mapper = PhysicsMapper()
    params = {
        'entropy': 0.7,
        'warp': 0.5,
        'luminosity': 0.9
    }
    
    latent_mod = mapper.map_to_latent(params)
    print(f"✅ Generated latent modification: shape {latent_mod.shape}")
    print(f"✅ Value range: [{latent_mod.min():.3f}, {latent_mod.max():.3f}]")
    
    print("\n✨ All tests passed!")