import wave
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import subprocess
import tempfile
from tqdm import tqdm

class AudioCompiler:
    def __init__(self, output_dir: str = "output/final"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio settings
        self.sample_rate = 24000
        self.channels = 1
        self.sample_width = 2  # 16-bit
        
        # Check if ffmpeg is available for advanced audio processing
        self.ffmpeg_available = self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available for audio processing"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            print("✅ FFmpeg detected - advanced audio processing available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ FFmpeg not found - using basic audio processing")
            return False
    
    def compile_audiobook(self, audio_files: List[str], book_name: str, 
                         add_silence: bool = True, normalize: bool = True,
                         mode: str = "multi", voice_name: Optional[str] = None) -> str:
        """Compile individual audio chunks into a complete audiobook
        
        Args:
            audio_files: List of audio file paths to compile
            book_name: Name of the book (can include extension)
            add_silence: Whether to add silence between chunks
            normalize: Whether to normalize audio levels
            mode: "multi" or "single" speaker mode
            voice_name: Voice name for single speaker mode (optional)
        """
        if not audio_files:
            raise ValueError("No audio files provided for compilation")
        
        # Clean book name - remove extension and path
        book_path = Path(book_name)
        clean_book_name = book_path.stem  # Gets filename without extension
        
        # Create safe filename based on mode
        safe_book_name = "".join(c for c in clean_book_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        if mode == "single" and voice_name:
            # Single Speaker: "Book Name - Single Speaker - Voice Name"
            safe_voice_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_filename = f"{safe_book_name} - Single Speaker - {safe_voice_name}.wav"
        else:
            # Multi Speaker: "Book Name - Multi Speaker"
            output_filename = f"{safe_book_name} - Multi Speaker.wav"
        
        output_file = self.output_dir / output_filename
        
        print(f"Compiling {len(audio_files)} audio chunks into: {output_file.name}")
        
        try:
            if self.ffmpeg_available and len(audio_files) > 1:
                return self._compile_with_ffmpeg(audio_files, output_file, add_silence, normalize)
            else:
                return self._compile_with_wave(audio_files, output_file, add_silence)
        
        except Exception as e:
            print(f"❌ Error compiling audiobook: {e}")
            # Try fallback method
            if self.ffmpeg_available:
                print("Trying fallback compilation with wave module...")
                return self._compile_with_wave(audio_files, output_file, add_silence)
            else:
                raise
    
    def _compile_with_wave(self, audio_files: List[str], output_file: Path, 
                          add_silence: bool = True) -> str:
        """Compile audio files using Python wave module (basic)"""
        silence_duration = 0.5  # seconds of silence between chunks
        silence_frames = int(self.sample_rate * silence_duration)
        silence_data = b'\x00\x00' * silence_frames
        
        with wave.open(str(output_file), 'wb') as output_wav:
            # Set output parameters
            output_wav.setnchannels(self.channels)
            output_wav.setsampwidth(self.sample_width)
            output_wav.setframerate(self.sample_rate)
            
            total_duration = 0
            
            # Create progress bar for compilation
            compilation_progress = tqdm(
                audio_files,
                desc="🔧 Compiling audio",
                unit="file",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
            
            for i, audio_file in enumerate(compilation_progress):
                compilation_progress.set_description(f"🔧 Compiling {i+1}/{len(audio_files)}")
                
                if not Path(audio_file).exists():
                    print(f"⚠️ Audio file not found: {audio_file}")
                    continue
                
                try:
                    with wave.open(audio_file, 'rb') as input_wav:
                        # Check format compatibility
                        input_channels = input_wav.getnchannels()
                        input_width = input_wav.getsampwidth()
                        input_rate = input_wav.getframerate()
                        
                        if (input_channels != self.channels or
                            input_width != self.sample_width or
                            input_rate != self.sample_rate):
                            print(f"⚠️ Audio format mismatch in {audio_file}")
                            print(f"   Expected: {self.channels}ch, {self.sample_width*8}bit, {self.sample_rate}Hz")
                            print(f"   Found: {input_channels}ch, {input_width*8}bit, {input_rate}Hz")
                            
                            # Try basic format conversion
                            frames = self._convert_audio_format(input_wav, input_channels, input_width, input_rate)
                            if frames:
                                output_wav.writeframes(frames)
                                duration = len(frames) // (self.channels * self.sample_width) / self.sample_rate
                                total_duration += duration
                                print(f"   ✅ Converted and added to compilation")
                            else:
                                print(f"   ❌ Conversion failed, skipping file")
                                continue
                        else:
                            # Compatible format, copy directly
                            frames = input_wav.readframes(input_wav.getnframes())
                            output_wav.writeframes(frames)
                            
                            # Calculate duration
                            duration = input_wav.getnframes() / input_wav.getframerate()
                            total_duration += duration
                        
                        # Add silence between chunks (except after the last one)
                        if add_silence and i < len(audio_files) - 1:
                            output_wav.writeframes(silence_data)
                            total_duration += silence_duration
                
                except Exception as e:
                    print(f"⚠️ Error processing {audio_file}: {e}")
                    continue
        
        # Close compilation progress bar
        compilation_progress.close()
        print(f"✅ Compiled audiobook: {total_duration:.2f}s total duration")
        return str(output_file)
    
    def _convert_audio_format(self, input_wav, input_channels, input_width, input_rate):
        """Convert audio to target format (basic conversion)"""
        try:
            import array
            import struct
            
            # Read all frames
            frames = input_wav.readframes(input_wav.getnframes())
            
            # For now, handle simple cases
            if input_width == self.sample_width and input_channels == self.channels:
                # Only sample rate conversion needed
                if input_rate != self.sample_rate:
                    # Simple sample rate conversion (basic)
                    return self._resample_audio(frames, input_rate, self.sample_rate, input_width)
                else:
                    return frames
            
            # For more complex conversions, skip for now
            print(f"   Complex format conversion not implemented")
            return None
            
        except Exception as e:
            print(f"   Error in format conversion: {e}")
            return None
    
    def _resample_audio(self, frames, input_rate, output_rate, sample_width):
        """Basic audio resampling (linear interpolation)"""
        try:
            if input_rate == output_rate:
                return frames
            
            # Convert bytes to samples
            if sample_width == 2:  # 16-bit
                fmt = '<h'  # little-endian signed short
            else:
                return None  # Only support 16-bit for now
            
            import struct
            samples = struct.unpack(f'<{len(frames)//sample_width}h', frames)
            
            # Calculate resampling ratio
            ratio = output_rate / input_rate
            output_length = int(len(samples) * ratio)
            
            # Simple linear interpolation resampling
            output_samples = []
            for i in range(output_length):
                # Map output index to input index
                input_index = i / ratio
                
                # Linear interpolation
                index_floor = int(input_index)
                index_ceil = min(index_floor + 1, len(samples) - 1)
                
                if index_floor == index_ceil:
                    sample = samples[index_floor]
                else:
                    # Interpolate between floor and ceil
                    frac = input_index - index_floor
                    sample = samples[index_floor] * (1 - frac) + samples[index_ceil] * frac
                
                output_samples.append(int(sample))
            
            # Convert back to bytes
            return struct.pack(f'<{len(output_samples)}h', *output_samples)
            
        except Exception as e:
            print(f"   Error in resampling: {e}")
            return None
    
    def _compile_with_ffmpeg(self, audio_files: List[str], output_file: Path,
                            add_silence: bool = True, normalize: bool = True) -> str:
        """Compile audio files using FFmpeg (advanced)"""
        
        # Create temporary file list for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            file_list_path = f.name
            
            for i, audio_file in enumerate(audio_files):
                if Path(audio_file).exists():
                    f.write(f"file '{Path(audio_file).absolute()}'\n")
                    
                    # Add silence between files
                    if add_silence and i < len(audio_files) - 1:
                        # Create a temporary silence file
                        silence_file = self._create_silence_file(0.5)  # 0.5 second silence
                        f.write(f"file '{silence_file}'\n")
        
        try:
            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-f', 'concat',
                '-safe', '0',
                '-i', file_list_path,
                '-c:a', 'pcm_s16le',  # 16-bit PCM
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', str(self.channels),  # Channels
            ]
            
            # Add normalization if requested
            if normalize:
                cmd.extend(['-filter:a', 'dynaudnorm'])
            
            cmd.append(str(output_file))
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Get duration info
            duration = self._get_audio_duration(str(output_file))
            print(f"✅ Compiled audiobook with FFmpeg: {duration:.2f}s total duration")
            
            return str(output_file)
        
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            raise
        
        finally:
            # Clean up temporary files
            try:
                os.unlink(file_list_path)
            except:
                pass
    
    def _create_silence_file(self, duration: float) -> str:
        """Create a temporary silence file"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        
        frames = int(self.sample_rate * duration)
        
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b'\x00\x00' * frames)
        
        return temp_file.name
    
    def _get_audio_duration(self, audio_file: str) -> float:
        """Get duration of audio file"""
        try:
            if self.ffmpeg_available:
                cmd = [
                    'ffprobe', '-v', 'quiet',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    audio_file
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return float(result.stdout.strip())
            else:
                # Fallback to wave module
                with wave.open(audio_file, 'rb') as wf:
                    return wf.getnframes() / wf.getframerate()
        except:
            return 0.0
    
    def create_chapters(self, audio_files: List[str], chapter_info: List[Dict[str, Any]], 
                       book_name: str) -> str:
        """Create audiobook with chapter markers (if ffmpeg available)"""
        if not self.ffmpeg_available:
            print("Chapter creation requires FFmpeg - falling back to regular compilation")
            return self.compile_audiobook(audio_files, book_name)
        
        # This would implement chapter creation using ffmpeg
        # For now, just compile normally
        return self.compile_audiobook(audio_files, book_name)
    
    def convert_format(self, input_file: str, output_format: str = 'mp3') -> str:
        """Convert audiobook to different format"""
        if not self.ffmpeg_available:
            raise ValueError("Format conversion requires FFmpeg")
        
        input_path = Path(input_file)
        output_path = input_path.with_suffix(f'.{output_format}')
        
        try:
            cmd = ['ffmpeg', '-y', '-i', str(input_path)]
            
            if output_format == 'mp3':
                cmd.extend(['-codec:a', 'libmp3lame', '-b:a', '128k'])
            elif output_format == 'm4a':
                cmd.extend(['-codec:a', 'aac', '-b:a', '128k'])
            elif output_format == 'ogg':
                cmd.extend(['-codec:a', 'libvorbis', '-q:a', '5'])
            
            cmd.append(str(output_path))
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            print(f"✅ Converted to {output_format}: {output_path}")
            return str(output_path)
        
        except subprocess.CalledProcessError as e:
            print(f"Conversion error: {e}")
            raise
    
    def get_compilation_stats(self, audio_files: List[str]) -> Dict[str, Any]:
        """Get statistics about audio files before compilation"""
        stats = {
            'total_files': len(audio_files),
            'total_duration': 0.0,
            'valid_files': 0,
            'invalid_files': 0,
            'file_sizes': [],
            'format_issues': []
        }
        
        for audio_file in audio_files:
            if not Path(audio_file).exists():
                stats['invalid_files'] += 1
                continue
            
            try:
                # Get file size
                file_size = Path(audio_file).stat().st_size
                stats['file_sizes'].append(file_size)
                
                # Get duration
                duration = self._get_audio_duration(audio_file)
                stats['total_duration'] += duration
                
                # Check format
                with wave.open(audio_file, 'rb') as wf:
                    if (wf.getnchannels() != self.channels or
                        wf.getsampwidth() != self.sample_width or
                        wf.getframerate() != self.sample_rate):
                        stats['format_issues'].append(audio_file)
                
                stats['valid_files'] += 1
                
            except Exception as e:
                stats['invalid_files'] += 1
                print(f"Error analyzing {audio_file}: {e}")
        
        stats['average_file_size'] = sum(stats['file_sizes']) / len(stats['file_sizes']) if stats['file_sizes'] else 0
        stats['total_size_mb'] = sum(stats['file_sizes']) / (1024 * 1024)
        
        return stats
    
    def cleanup_temp_files(self):
        """Clean up any temporary files created during compilation"""
        # Clean up any temporary silence files
        temp_dir = Path(tempfile.gettempdir())
        for temp_file in temp_dir.glob("tmp*.wav"):
            try:
                temp_file.unlink()
            except:
                pass