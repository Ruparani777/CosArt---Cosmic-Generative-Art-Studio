import React, { useState, useEffect, useRef } from 'react';
import { Camera, Sparkles, Orbit, Wand2, Layers, Download, Play, Pause } from 'lucide-react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Text } from '@react-three/drei';

const CosArtApp = () => {
  const [activeTab, setActiveTab] = useState('generate');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [progress, setProgress] = useState(0);
  const [seed, setSeed] = useState(Math.floor(Math.random() * 1000000));
  
  // Physics parameters (Cosmic Controls)
  const [physicsParams, setPhysicsParams] = useState({
    entropy: 0.5,
    warp: 0.5,
    luminosity: 0.7,
    cosmic_flow: 0.5,
    pattern_collapse: 0.3,
    attraction: 0.5,
    uncertainty: 0.2,
    spectral_shift: 0.5
  });
  
  const [selectedPreset, setSelectedPreset] = useState('nebula');
  const [resolution, setResolution] = useState(512);
  
  // Universe Mode state
  const [universeData, setUniverseData] = useState([]);
  const [selectedArtwork, setSelectedArtwork] = useState(null);
  const [isExploring, setIsExploring] = useState(false);

  // Simulate universe data (replace with API call)
  useEffect(() => {
    if (activeTab === 'universe' && universeData.length === 0) {
      // Generate mock universe points
      const points = [];
      for (let i = 0; i < 1000; i++) {
        points.push({
          id: i,
          position: [
            (Math.random() - 0.5) * 100,
            (Math.random() - 0.5) * 100,
            (Math.random() - 0.5) * 100
          ],
          color: `hsl(${Math.random() * 360}, 70%, 50%)`,
          seed: Math.floor(Math.random() * 1000000)
        });
      }
      setUniverseData(points);
    }
  }, [activeTab, universeData.length]);

  // Universe Point Component
  function UniversePoint({ point, onClick }) {
    const meshRef = useRef();
    
    useFrame((state) => {
      if (meshRef.current) {
        meshRef.current.rotation.x += 0.005;
        meshRef.current.rotation.y += 0.005;
      }
    });

    return (
      <mesh
        ref={meshRef}
        position={point.position}
        onClick={() => onClick(point)}
        onPointerOver={() => document.body.style.cursor = 'pointer'}
        onPointerOut={() => document.body.style.cursor = 'auto'}
      >
        <sphereGeometry args={[0.5, 8, 8]} />
        <meshBasicMaterial color={point.color} />
      </mesh>
    );
  }

  // Universe Scene Component
  function UniverseScene() {
    const handlePointClick = (point) => {
      setSelectedArtwork(point);
      // Simulate generation
      setIsGenerating(true);
      setProgress(0);
      const interval = setInterval(() => {
        setProgress(p => {
          if (p >= 100) {
            clearInterval(interval);
            setIsGenerating(false);
            setGeneratedImage(`https://picsum.photos/seed/${point.seed}/512/512`);
            setActiveTab('generate'); // Switch to generate tab to show result
            return 100;
          }
          return p + 10;
        });
      }, 200);
    };

    return (
      <>
        <Stars radius={300} depth={50} count={5000} factor={4} saturation={0} fade />
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        {universeData.map((point) => (
          <UniversePoint key={point.id} point={point} onClick={handlePointClick} />
        ))}
        <Text
          position={[0, 60, 0]}
          fontSize={8}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          Cosmic Universe Explorer
        </Text>
        <Text
          position={[0, 50, 0]}
          fontSize={3}
          color="#a855f7"
          anchorX="center"
          anchorY="middle"
        >
          Click on stars to generate art from that cosmic location
        </Text>
      </>
    );
  }
  
  // Cosmic presets
  const presets = {
    nebula: { name: '🌌 Nebula', color: 'from-purple-500 to-pink-500' },
    black_hole: { name: '🕳️ Black Hole', color: 'from-gray-900 to-purple-900' },
    galaxy: { name: '🌀 Galaxy', color: 'from-blue-500 to-purple-500' },
    star_field: { name: '✨ Star Field', color: 'from-indigo-900 to-blue-500' },
    dark_matter: { name: '👻 Dark Matter', color: 'from-purple-900 to-black' },
    cosmic_web: { name: '🕸️ Cosmic Web', color: 'from-cyan-500 to-purple-500' }
  };

  // Simulate generation
  const handleGenerate = async () => {
    setIsGenerating(true);
    setProgress(0);
    
    // Simulate progress
    const interval = setInterval(() => {
      setProgress(p => {
        if (p >= 100) {
          clearInterval(interval);
          setIsGenerating(false);
          // Simulate generated image
          setGeneratedImage(`https://picsum.photos/seed/${seed}/512/512`);
          return 100;
        }
        return p + 5;
      });
    }, 200);
  };

  const updatePhysicsParam = (param, value) => {
    setPhysicsParams(prev => ({ ...prev, [param]: value }));
  };

  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 1000000));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-black text-white">
      {/* Header */}
      <header className="border-b border-purple-500/30 bg-black/30 backdrop-blur-lg">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                <Sparkles className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  CosArt
                </h1>
                <p className="text-xs text-gray-400">Where Art, Math & Universe Meet</p>
              </div>
            </div>
            
            <nav className="flex gap-4">
              <button
                onClick={() => setActiveTab('generate')}
                className={`px-4 py-2 rounded-lg transition-all ${
                  activeTab === 'generate'
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Wand2 className="w-4 h-4 inline mr-2" />
                Generate
              </button>
              <button
                onClick={() => setActiveTab('universe')}
                className={`px-4 py-2 rounded-lg transition-all ${
                  activeTab === 'universe'
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Orbit className="w-4 h-4 inline mr-2" />
                Universe Mode
              </button>
              <button
                onClick={() => setActiveTab('gallery')}
                className={`px-4 py-2 rounded-lg transition-all ${
                  activeTab === 'gallery'
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Layers className="w-4 h-4 inline mr-2" />
                Galaxy
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {activeTab === 'generate' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Controls Panel */}
            <div className="lg:col-span-1 space-y-6">
              {/* Cosmic Presets */}
              <div className="bg-black/40 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-purple-400" />
                  Cosmic Presets
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(presets).map(([key, preset]) => (
                    <button
                      key={key}
                      onClick={() => setSelectedPreset(key)}
                      className={`p-3 rounded-lg text-sm transition-all ${
                        selectedPreset === key
                          ? `bg-gradient-to-r ${preset.color} text-white shadow-lg`
                          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                      }`}
                    >
                      {preset.name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Big Bang Seed */}
              <div className="bg-black/40 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
                <h3 className="text-lg font-semibold mb-4">🌟 Big Bang Seed</h3>
                <div className="flex gap-2">
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value))}
                    className="flex-1 bg-gray-800/50 border border-purple-500/30 rounded-lg px-4 py-2 text-white"
                  />
                  <button
                    onClick={randomizeSeed}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
                  >
                    🎲
                  </button>
                </div>
              </div>

              {/* Physics Controls */}
              <div className="bg-black/40 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30 max-h-96 overflow-y-auto">
                <h3 className="text-lg font-semibold mb-4">⚛️ Physics Controls</h3>
                <div className="space-y-4">
                  {Object.entries(physicsParams).map(([param, value]) => (
                    <div key={param}>
                      <div className="flex justify-between mb-2">
                        <label className="text-sm text-gray-300 capitalize">
                          {param.replace('_', ' ')}
                        </label>
                        <span className="text-sm text-purple-400">{value.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={value}
                        onChange={(e) => updatePhysicsParam(param, parseFloat(e.target.value))}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                      />
                    </div>
                  ))}
                </div>
              </div>

              {/* Resolution */}
              <div className="bg-black/40 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
                <h3 className="text-lg font-semibold mb-4">📐 Resolution</h3>
                <select
                  value={resolution}
                  onChange={(e) => setResolution(parseInt(e.target.value))}
                  className="w-full bg-gray-800/50 border border-purple-500/30 rounded-lg px-4 py-2 text-white"
                >
                  <option value="512">512x512 (Fast)</option>
                  <option value="1024">1024x1024 (Standard)</option>
                  <option value="2048">2048x2048 (High)</option>
                </select>
              </div>

              {/* Generate Button */}
              <button
                onClick={handleGenerate}
                disabled={isGenerating}
                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-bold py-4 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-purple-500/50"
              >
                {isGenerating ? (
                  <span className="flex items-center justify-center gap-2">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Generating Universe... {progress}%
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-2">
                    <Sparkles className="w-5 h-5" />
                    Create Cosmic Art
                  </span>
                )}
              </button>
            </div>

            {/* Canvas Area */}
            <div className="lg:col-span-2">
              <div className="bg-black/40 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30 h-full">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">🎨 Cosmic Canvas</h3>
                  {generatedImage && (
                    <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors flex items-center gap-2">
                      <Download className="w-4 h-4" />
                      Export
                    </button>
                  )}
                </div>
                
                <div className="aspect-square bg-gradient-to-br from-gray-900 to-purple-900/30 rounded-lg flex items-center justify-center overflow-hidden border border-purple-500/20">
                  {isGenerating ? (
                    <div className="text-center">
                      <div className="w-20 h-20 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                      <p className="text-gray-400">Evolving your universe...</p>
                      <div className="w-64 bg-gray-800 rounded-full h-2 mt-4">
                        <div
                          className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                    </div>
                  ) : generatedImage ? (
                    <img
                      src={generatedImage}
                      alt="Generated cosmic art"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="text-center text-gray-500">
                      <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p>Your cosmic masterpiece will appear here</p>
                      <p className="text-sm mt-2">Adjust physics controls and click generate</p>
                    </div>
                  )}
                </div>

                {generatedImage && (
                  <div className="mt-4 p-4 bg-gray-900/50 rounded-lg border border-purple-500/20">
                    <h4 className="text-sm font-semibold mb-2 text-purple-400">Generation Metadata</h4>
                    <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
                      <div>Seed: {seed}</div>
                      <div>Preset: {presets[selectedPreset].name}</div>
                      <div>Resolution: {resolution}x{resolution}</div>
                      <div>Entropy: {physicsParams.entropy.toFixed(2)}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'universe' && (
          <div className="bg-black/40 backdrop-blur-lg rounded-xl p-8 border border-purple-500/30">
            <div className="mb-6 text-center">
              <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Universe Mode
              </h2>
              <p className="text-gray-400 mb-4 max-w-2xl mx-auto">
                Navigate through a 3D latent space where each point represents a unique artwork. 
                Travel through galaxies of creativity and discover hidden constellations.
              </p>
              <div className="flex justify-center gap-4 mb-4">
                <button 
                  onClick={() => setIsExploring(!isExploring)}
                  className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-bold rounded-xl transition-all shadow-lg"
                >
                  {isExploring ? <Pause className="w-5 h-5 inline mr-2" /> : <Play className="w-5 h-5 inline mr-2" />}
                  {isExploring ? 'Pause Exploration' : 'Start Exploration'}
                </button>
              </div>
            </div>
            
            <div className="h-96 bg-black/50 rounded-lg overflow-hidden">
              <Canvas camera={{ position: [0, 0, 50], fov: 75 }}>
                <UniverseScene />
              </Canvas>
            </div>
            
            <div className="mt-6 text-center text-sm text-gray-400">
              <p>🖱️ Click and drag to navigate • 🔍 Scroll to zoom • ⭐ Click stars to generate art</p>
              {selectedArtwork && (
                <p className="mt-2 text-purple-400">
                  Selected cosmic location: Seed {selectedArtwork.seed}
                </p>
              )}
            </div>
          </div>
        )}

        {activeTab === 'gallery' && (
          <div className="bg-black/40 backdrop-blur-lg rounded-xl p-8 border border-purple-500/30">
            <h2 className="text-2xl font-bold mb-6">🌌 Your Personal Galaxy</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                <div
                  key={i}
                  className="aspect-square bg-gradient-to-br from-purple-900/30 to-pink-900/30 rounded-lg border border-purple-500/20 flex items-center justify-center hover:border-purple-500 transition-all cursor-pointer"
                >
                  <span className="text-gray-600">Artwork {i}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-purple-500/30 bg-black/30 backdrop-blur-lg mt-20">
        <div className="container mx-auto px-6 py-4 text-center text-gray-400 text-sm">
          <p>"Where algorithms dream in constellations and artists become cosmic architects."</p>
        </div>
      </footer>
    </div>
  );
};

export default CosArtApp;