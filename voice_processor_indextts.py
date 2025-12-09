"""
IndexTTS Voice Processor
Handles text-to-speech generation using IndexTTS with one-shot voice cloning.
Supports both GPU and CPU modes for flexibility.
"""

import os
import sys
import wave
import time
import torch
import torchaudio
import librosa
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import threading
import hashlib
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Emotion vector presets (8 floats: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm)
EMOTION_VECTORS = {
    "neutral": None,  # No emotion override, use voice sample's natural emotion
    "calm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Pure calm
    "happy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],  # Happy with slight surprise
    "sad": [0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0],  # Sad with melancholic
    "angry": [0.0, 1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],  # Angry with disgust
}

# Add index-tts to Python path
INDEX_TTS_PATH = Path(__file__).parent / "index-tts"
if INDEX_TTS_PATH.exists():
    sys.path.insert(0, str(INDEX_TTS_PATH))

def check_device_availability():
    """Detect available compute device (CUDA, MPS, XPU, or CPU)"""
    if torch.cuda.is_available():
        return "cuda:0", True
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", True
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps", False  # MPS doesn't benefit from FP16
    else:
        return "cpu", False

# Detect device and FP16 capability
DEVICE, CAN_USE_FP16 = check_device_availability()
print(f"🎮 Using device: {DEVICE}")
if CAN_USE_FP16:
    print("⚡ FP16 acceleration available")

# Import IndexTTS2
try:
    from indextts.infer_v2 import IndexTTS2  # type: ignore
    INDEXTTS_AVAILABLE = True
    print("✅ IndexTTS2 imported successfully")
except Exception as e:
    print(f"❌ IndexTTS2 not available: {e}")
    print("   Please ensure index-tts is properly installed")
    print("   Run: python3 setup_indextts.py")
    INDEXTTS_AVAILABLE = False
    # Create a dummy class to prevent errors
    class IndexTTS2:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("IndexTTS2 not available - run setup_indextts.py first")


class VoiceProcessorIndexTTS:
    """Voice processor using IndexTTS for one-shot voice cloning"""
    
    def __init__(self, 
                 model_dir: str = "index-tts/checkpoints",
                 config_path: str = "index-tts/checkpoints/config.yaml",
                 use_fp16: bool = None,
                 device: str = None,
                 voice_samples_dir: str = "voice_samples"):
        """
        Initialize the IndexTTS voice processor
        
        Args:
            model_dir: Path to IndexTTS model directory
            config_path: Path to config.yaml
            use_fp16: Use FP16 for faster inference (auto-detect if None)
            device: Device to use (auto-detect if None)
            voice_samples_dir: Directory containing custom voice samples
        """
        self.model_dir = Path(model_dir)
        self.config_path = Path(config_path)
        self.voice_samples_dir = Path(voice_samples_dir)
        
        # Device setup
        self.device = device if device else DEVICE
        self.use_fp16 = use_fp16 if use_fp16 is not None else (CAN_USE_FP16 and self.device.startswith("cuda"))
        self.indextts_available = INDEXTTS_AVAILABLE
        
        # Thread safety
        self.model_lock = threading.Lock()
        
        # Output directories
        self.chunk_dir = Path("output/chunks")
        self.final_dir = Path("output/final")
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        
        # Voice samples cache
        self.voice_samples_cache = {}
        self.available_voices = []
        
        # Model instance
        self.tts_model = None
        
        print(f"\n{'='*60}")
        print(f"IndexTTS Voice Processor Initialization")
        print(f"{'='*60}")
        print(f"📁 Model directory: {self.model_dir}")
        print(f"📄 Config path: {self.config_path}")
        print(f"🎮 Device: {self.device}")
        print(f"⚡ FP16 enabled: {self.use_fp16}")
        print(f"🎤 Voice samples directory: {self.voice_samples_dir}")
        print(f"🔧 IndexTTS available: {self.indextts_available}")
        print(f"{'='*60}\n")
        
        # Create voice samples directory if it doesn't exist
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan for available voice samples
        self._scan_voice_samples()
        
        # Load the model
        if self.indextts_available:
            self._load_model()
        else:
            print("⚠️  IndexTTS not available - running in placeholder mode")
            print("    Install IndexTTS and download models to enable TTS generation")
    
    def _scan_voice_samples(self):
        """Scan voice_samples directory for available voice files"""
        print("🔍 Scanning for voice samples...")
        
        if not self.voice_samples_dir.exists():
            print(f"   Voice samples directory not found: {self.voice_samples_dir}")
            return
        
        # Supported audio formats
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        voice_files = []
        for ext in audio_extensions:
            voice_files.extend(self.voice_samples_dir.glob(f"*{ext}"))
        
        self.available_voices = []
        for voice_file in sorted(voice_files):
            voice_name = voice_file.stem  # Filename without extension
            self.available_voices.append(voice_name)
            self.voice_samples_cache[voice_name] = str(voice_file)
            print(f"   ✓ Found voice: {voice_name} ({voice_file.name})")
        
        if not self.available_voices:
            print(f"   ⚠️  No voice samples found in {self.voice_samples_dir}")
            print(f"       Add .wav or .mp3 files to use custom voices")
        else:
            print(f"   📊 Total voices available: {len(self.available_voices)}")
    
    def _load_model(self):
        """Load the IndexTTS2 model"""
        if not self.model_dir.exists():
            print(f"❌ Model directory not found: {self.model_dir}")
            print(f"   Run setup script to download models")
            self.tts_model = None
            return
        
        if not self.config_path.exists():
            print(f"❌ Config file not found: {self.config_path}")
            self.tts_model = None
            return
        
        try:
            print(f"📦 Loading IndexTTS2 model...")
            print(f"   This may take a minute on first run...")
            
            with self.model_lock:
                self.tts_model = IndexTTS2(
                    cfg_path=str(self.config_path),
                    model_dir=str(self.model_dir),
                    use_fp16=self.use_fp16,
                    device=self.device,
                    use_cuda_kernel=self.device.startswith("cuda"),
                    use_deepspeed=False  # Can be enabled for potential speed improvements
                )
            
            print(f"✅ IndexTTS2 model loaded successfully!")
            print(f"   Model version: {self.tts_model.model_version if hasattr(self.tts_model, 'model_version') else 'IndexTTS-2'}")
            
        except Exception as e:
            print(f"❌ Failed to load IndexTTS2 model: {e}")
            print(f"   Check that models are downloaded and config is correct")
            self.tts_model = None
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voice names"""
        return self.available_voices.copy()
    
    def get_voice_sample_path(self, voice_name: str) -> Optional[str]:
        """Get the file path for a voice sample"""
        return self.voice_samples_cache.get(voice_name)
    
    def validate_voice_sample(self, voice_path: str) -> Tuple[bool, str]:
        """
        Validate a voice sample file
        
        Returns:
            (is_valid, message)
        """
        voice_file = Path(voice_path)
        
        if not voice_file.exists():
            return False, f"File not found: {voice_path}"
        
        try:
            # Load audio to check
            audio, sr = librosa.load(voice_path, sr=None, mono=False)
            
            # Check duration
            duration = librosa.get_duration(y=audio, sr=sr)
            if duration < 3.0:
                return False, f"Audio too short: {duration:.1f}s (minimum 3s recommended)"
            if duration > 60.0:
                return False, f"Audio too long: {duration:.1f}s (maximum 60s recommended)"
            
            # Check sample rate
            if sr < 16000:
                return False, f"Sample rate too low: {sr}Hz (minimum 16kHz recommended)"
            
            return True, f"Valid voice sample: {duration:.1f}s at {sr}Hz"
            
        except Exception as e:
            return False, f"Error loading audio: {str(e)}"
    
    def generate_chunk_audio(self, 
                            text: str, 
                            voice_name: str, 
                            chunk_id: int,
                            book_name: str = "audiobook",
                            generation_kwargs: Optional[Dict[str, Any]] = None,
                            emotion_type: str = "neutral",
                            emotion_alpha: float = 1.0) -> Optional[str]:
        """
        Generate audio for a text chunk using specified voice
        
        Args:
            text: Text to synthesize
            voice_name: Name of voice sample to use
            chunk_id: Chunk number for output naming
            book_name: Book name for output naming
            generation_kwargs: Additional generation parameters
            emotion_type: Emotion to apply (neutral, calm, happy, sad, angry)
            emotion_alpha: Emotion intensity 0.0-1.0 (0=none, 1=full)
        
        Returns:
            Path to generated audio file, or None if failed
        """
        if not text.strip():
            print(f"⚠️  Empty text for chunk {chunk_id}, skipping")
            return None
        
        # Get voice sample path
        voice_sample_path = self.get_voice_sample_path(voice_name)
        if not voice_sample_path:
            print(f"⚠️  Voice '{voice_name}' not found, using fallback")
            # Try to find any available voice as fallback
            if self.available_voices:
                voice_name = self.available_voices[0]
                voice_sample_path = self.get_voice_sample_path(voice_name)
                print(f"   Using fallback voice: {voice_name}")
            else:
                return self._create_placeholder_audio(text, chunk_id, book_name, voice_name)
        
        # Generate output path
        safe_book_name = "".join(c for c in book_name if c.isalnum() or c in (' ', '_', '-')).strip()
        output_file = self.chunk_dir / f"{safe_book_name}_chunk_{chunk_id:04d}_{voice_name}.wav"
        
        # Check if already generated (caching)
        if output_file.exists():
            print(f"📦 Using cached audio for chunk {chunk_id}")
            return str(output_file)
        
        # Generate audio
        if self.tts_model is None:
            print(f"⚠️  TTS model not available, creating placeholder")
            return self._create_placeholder_audio(text, chunk_id, book_name, voice_name)
        
        try:
            print(f"🎵 Generating audio for chunk {chunk_id} with voice '{voice_name}'...")
            print(f"   Text: {text[:100]}..." if len(text) > 100 else f"   Text: {text}")
            
            # Get emotion vector based on type
            emotion_vector = EMOTION_VECTORS.get(emotion_type, None)
            if emotion_vector and emotion_alpha > 0.0:
                print(f"   🎭 Applying emotion: {emotion_type} (intensity: {emotion_alpha:.1f})")
            
            with self.model_lock:
                # Generate audio using IndexTTS2
                # Newer IndexTTS2.infer() requires an output_path argument and
                # may either write the file directly or return (wav, sr).
                # Allow optional generation overrides (e.g., diffusion_steps) to be passed
                gen_kwargs = generation_kwargs.copy() if generation_kwargs else {}
                result = self.tts_model.infer(
                    spk_audio_prompt=voice_sample_path,
                    text=text,
                    output_path=str(output_file),
                    emo_audio_prompt=voice_sample_path,  # Use same sample for emotion reference
                    emo_vector=emotion_vector,  # Apply emotion preset
                    emo_alpha=emotion_alpha,  # Emotion intensity
                    verbose=False,
                    **gen_kwargs,
                )

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # IndexTTS may either return (wav, sr) or write the file to output_path.
            # If it returned audio data, save it. Otherwise assume the file was written.
            try:
                if isinstance(result, tuple) and len(result) == 2:
                    wav, sr = result
                    # If wav is a numpy array or torch tensor, save via torchaudio
                    if isinstance(wav, np.ndarray):
                        tensor = torch.from_numpy(wav)
                    elif isinstance(wav, torch.Tensor):
                        tensor = wav
                    else:
                        # Unknown type, attempt to convert
                        tensor = torch.tensor(wav)

                    # Ensure mono channel first dimension for torchaudio (1, N)
                    if tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.dim() == 2 and tensor.shape[0] > 1:
                        # If multi-channel, keep as-is
                        pass

                    torchaudio.save(str(output_file), tensor, int(sr))
                else:
                    # Assume infer wrote the file to output_path
                    if not output_file.exists():
                        raise RuntimeError("IndexTTS infer did not return audio and did not write output file")
            except Exception as save_exc:
                print(f"❌ Failed to save generated audio: {save_exc}")
                raise
            
            print(f"✅ Generated audio saved to: {output_file.name}")
            return str(output_file)
            
        except Exception as e:
            print(f"❌ Error generating audio for chunk {chunk_id}: {e}")
            print(f"   Creating placeholder audio instead")
            return self._create_placeholder_audio(text, chunk_id, book_name, voice_name)
    
    def _create_placeholder_audio(self, 
                                  text: str, 
                                  chunk_id: int, 
                                  book_name: str,
                                  voice_name: str) -> Optional[str]:
        """Create a placeholder audio file when TTS is not available"""
        try:
            # Calculate duration based on text length (approximate speaking rate)
            duration = max(2.0, len(text) / 100.0)  # ~100 chars per second
            sample_rate = 24000  # Match expected format
            channels = 1  # Mono
            sample_width = 2  # 16-bit
            samples = int(duration * sample_rate)
            
            # Generate output path
            safe_book_name = "".join(c for c in book_name if c.isalnum() or c in (' ', '_', '-')).strip()
            output_file = self.chunk_dir / f"{safe_book_name}_chunk_{chunk_id:04d}_{voice_name}.wav"
            
            # Create the output directory
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a WAV file with silence
            with wave.open(str(output_file), 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                
                # Write silence (zeros)
                silence_data = b'\x00\x00' * samples
                wav_file.writeframes(silence_data)
            
            print(f"📝 Created placeholder audio: {output_file.name}")
            return str(output_file)
            
        except Exception as e:
            print(f"❌ Error creating placeholder audio: {e}")
            return None
    
    def cleanup_cache(self):
        """Clean up cached voice samples and model"""
        self.voice_samples_cache.clear()
        if self.tts_model is not None:
            del self.tts_model
            self.tts_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("🧹 Voice processor cache cleaned")
    
    def reload_voice_samples(self):
        """Reload voice samples from directory"""
        self._scan_voice_samples()
        print(f"🔄 Voice samples reloaded: {len(self.available_voices)} voices available")

    def save_voice_sample(self, uploaded_file_path: str, original_filename: Optional[str] = None) -> Tuple[bool, str]:
        """Save an uploaded voice sample into the voice_samples directory.

        Args:
            uploaded_file_path: Path to the temporary uploaded file (from Gradio)
            original_filename: Optional original filename to use; falls back to basename of uploaded_file_path

        Returns:
            (success, message)
        """
        try:
            src = Path(uploaded_file_path)
            if not src.exists():
                return False, f"Uploaded file not found: {uploaded_file_path}"

            # Use original filename if provided, otherwise the uploaded file's name
            dest_name = original_filename if original_filename else src.name
            # Sanitize filename
            dest_name = "".join(c for c in dest_name if c.isalnum() or c in (' ', '.', '_', '-')).strip()
            dest_path = self.voice_samples_dir / dest_name

            # If a file with same name exists, append a numeric suffix
            if dest_path.exists():
                base = dest_path.stem
                ext = dest_path.suffix
                i = 1
                while True:
                    candidate = self.voice_samples_dir / f"{base}_{i}{ext}"
                    if not candidate.exists():
                        dest_path = candidate
                        break
                    i += 1

            # Copy file into voice_samples directory
            shutil.copy2(str(src), str(dest_path))

            # Validate sample
            valid, msg = self.validate_voice_sample(str(dest_path))
            if not valid:
                # Remove invalid file
                try:
                    dest_path.unlink()
                except:
                    pass
                return False, f"Invalid voice sample: {msg}"

            # Re-scan to update cache
            self._scan_voice_samples()
            return True, f"Voice sample '{dest_path.name}' uploaded successfully ({msg})"

        except Exception as e:
            return False, f"Failed to save voice sample: {str(e)}"

    def delete_voice_sample(self, voice_name: str) -> Tuple[bool, str]:
        """Delete a voice sample by voice name (stem without extension).

        Returns:
            (success, message)
        """
        try:
            path = self.get_voice_sample_path(voice_name)
            if not path:
                return False, f"Voice '{voice_name}' not found"

            p = Path(path)
            if p.exists():
                p.unlink()
            # Update cache and available voices
            if voice_name in self.voice_samples_cache:
                del self.voice_samples_cache[voice_name]
            self._scan_voice_samples()
            return True, f"Voice sample '{voice_name}' deleted"
        except Exception as e:
            return False, f"Failed to delete voice sample '{voice_name}': {str(e)}"


# For backward compatibility
VoiceProcessor = VoiceProcessorIndexTTS
