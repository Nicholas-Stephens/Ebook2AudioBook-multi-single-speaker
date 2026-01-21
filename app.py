import gradio as gr
import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import threading
import queue
import time
from dataclasses import dataclass
from tqdm import tqdm

from book_analyzer import BookAnalyzer
# Use IndexTTS voice processor instead of Orpheus
try:
    from voice_processor_indextts import VoiceProcessorIndexTTS as VoiceProcessor
    print("✅ Using IndexTTS Voice Processor")
except ImportError:
    print("⚠️  IndexTTS not available, falling back to old processor")
    from voice_processor import VoiceProcessor
from audio_compiler import AudioCompiler

# Configuration: Books directory (can be set via environment variable for Docker)
BOOKS_DIR = os.environ.get("BOOKS_DIR", "books")

@dataclass
class GenerationProgress:
    total_chunks: int = 0
    processed_chunks: int = 0
    current_task: str = ""
    status: str = "idle"
    error: Optional[str] = None
    start_time: Optional[float] = None

class AudiobookGenerator:
    def __init__(self):
        self.book_analyzer = BookAnalyzer()
        self.voice_processor = VoiceProcessor()
        # Create separate audio compilers for multi and single speaker modes
        self.audio_compiler_multi = AudioCompiler(output_dir="output/multi_final")
        self.audio_compiler_single = AudioCompiler(output_dir="output/single_final")
        self.progress = GenerationProgress()
        self.generation_queue = queue.Queue()
        
        # Available Orpheus voices
        self.available_voices = [
            "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"
        ]
        
        # Ensure output directories exist
        Path("output").mkdir(exist_ok=True)
        Path(BOOKS_DIR).mkdir(parents=True, exist_ok=True)
        print(f"📂 Books directory: {Path(BOOKS_DIR).absolute()}")
    
    def _resolve_voice_name(self, speaker_name: str) -> Optional[str]:
        """Resolve a speaker name to an available voice sample using fuzzy matching
        
        Args:
            speaker_name: Name of the speaker from book analysis
            
        Returns:
            Matching voice name from available voices, or None if no match
        """
        if not speaker_name or speaker_name == "Narrator":
            return None
        
        # Get available voices from voice processor
        available_voices = self.voice_processor.available_voices
        
        # Try exact match first (case-insensitive)
        speaker_lower = speaker_name.lower()
        for voice_name in available_voices:
            if speaker_lower == voice_name.lower():
                return voice_name
        
        # Try partial match - check if speaker name is in voice name
        for voice_name in available_voices:
            if speaker_lower in voice_name.lower():
                return voice_name
        
        # Try reverse - check if voice name is in speaker name
        for voice_name in available_voices:
            voice_base = voice_name.split(' - ')[0].lower()  # Handle "Name - Game" format
            if voice_base in speaker_lower:
                return voice_name
        
        return None

    def _get_voice_for_speaker(self, speaker_name: str, voice_assignments: Dict[str, str]) -> Optional[str]:
        """Resolve the final voice name to use for a speaker.

        Resolution order:
        1. Exact match in voice_assignments (case-sensitive keys as created from analysis)
        2. Case-insensitive exact match
        3. Partial / containment match against assignment keys
        4. Use the 'Narrator' assignment as a fallback for minor/unresolved speakers
        5. Try to auto-resolve by matching speaker name to available voice sample names

        Returns:
            Resolved voice name (string) or None if not found
        """
        # 1) Exact key match
        if speaker_name in voice_assignments and voice_assignments[speaker_name] and voice_assignments[speaker_name] != "(Not Assigned)":
            return voice_assignments[speaker_name]

        # 2) Case-insensitive exact match
        speaker_lower = speaker_name.lower() if speaker_name else ""
        for key in voice_assignments.keys():
            if key.lower() == speaker_lower and voice_assignments.get(key) and voice_assignments.get(key) != "(Not Assigned)":
                print(f"   🔁 Matched speaker '{speaker_name}' -> assignment key '{key}' (case-insensitive)")
                return voice_assignments.get(key)

        # 3) Partial/containment match against keys (e.g., 'maeve' in 'Maeve')
        for key in voice_assignments.keys():
            if speaker_lower in key.lower() and voice_assignments.get(key) and voice_assignments.get(key) != "(Not Assigned)":
                print(f"   🔁 Partial match: speaker '{speaker_name}' -> assignment key '{key}'")
                return voice_assignments.get(key)

        # 4) Narrator fallback if present
        narrator_voice = voice_assignments.get('Narrator') or voice_assignments.get('narrator')
        if narrator_voice and narrator_voice != "(Not Assigned)":
            # Use narrator for empty/unknown speakers and minor characters
            if not speaker_name or speaker_name.lower() == 'narrator':
                return narrator_voice

        # 5) Try to auto-resolve by matching speaker name to available voice samples
        auto_resolved = self._resolve_voice_name(speaker_name)
        if auto_resolved:
            print(f"   🔍 Auto-resolved speaker '{speaker_name}' -> voice sample '{auto_resolved}'")
            return auto_resolved

        # No mapping found
        return None
        
    def get_book_files(self) -> List[str]:
        """Get list of available books from the books folder (recursively scans subfolders)"""
        book_folder = Path(BOOKS_DIR)
        supported_formats = ['.txt', '.pdf', '.epub', '.docx', '.odt', '.rtf', '.doc']
        books = []
        
        if book_folder.exists():
            # Use rglob to recursively find all book files in subfolders
            for file in book_folder.rglob("*"):
                if file.is_file() and file.suffix.lower() in supported_formats:
                    # Store relative path from books folder (e.g., "Stephen King/IT.epub")
                    relative_path = file.relative_to(book_folder)
                    books.append(str(relative_path))
        
        return sorted(books)
    
    def analyze_book_speakers(self, book_path: str) -> Tuple[Dict[str, Any], str]:
        """Analyze book and extract speaker information"""
        try:
            self.progress.status = "analyzing"
            self.progress.current_task = "Extracting text from book..."
            print(f"📖 Extracting text from: {book_path}")
            
            # Extract text from book
            full_text = self.book_analyzer.extract_text(book_path)
            print(f"📝 Extracted {len(full_text):,} characters from book")
            
            self.progress.current_task = "Analyzing speakers with Gemini..."
            print("🤖 Analyzing characters and speakers with Gemini AI...")
            
            # Get speaker analysis from Gemini
            speaker_analysis = self.book_analyzer.analyze_speakers(full_text)
            print(f"🎭 Found {len(speaker_analysis)} speakers in the book")
            
            self.progress.current_task = "Chunking text for processing..."
            print("✂️ Chunking text for audio generation...")
            
            # Chunk the text intelligently
            chunks = self.book_analyzer.chunk_text(full_text, speaker_analysis)
            self.progress.total_chunks = len(chunks)
            print(f"📄 Created {len(chunks)} text chunks for processing")
            
            print("✅ Book analysis complete!")
            return speaker_analysis, "Analysis complete!"
            
        except Exception as e:
            self.progress.error = str(e)
            return {}, f"Error analyzing book: {str(e)}"
    
    def generate_audiobook(self, book_file: str, voice_assignments: Dict[str, str], fast_preview: bool = False, is_single_speaker: bool = False, emotion_type: str = "neutral", emotion_alpha: float = 1.0) -> str:
        """Generate the complete audiobook
        
        Args:
            book_file: Name of book file
            voice_assignments: Dict mapping speaker names to voice names
            fast_preview: If True, use lower quality settings for faster CPU generation (not recommended for final output)
            is_single_speaker: If True, use single_final output directory, otherwise use multi_final
            emotion_type: Emotion to apply (neutral, calm, happy, sad, angry)
            emotion_alpha: Emotion intensity 0.0-1.0
        """
        try:
            self.progress.status = "generating"
            self.progress.processed_chunks = 0
            self.progress.current_task = "Starting audiobook generation..."
            self.progress.start_time = time.time()  # Track start time for ETA
            
            book_path = Path("books") / book_file
            
            # Get book analysis and chunks
            speaker_analysis = self.book_analyzer.get_cached_analysis(book_path)
            chunks = self.book_analyzer.get_cached_chunks(book_path)
            
            if not chunks:
                return "Error: Book analysis not found. Please analyze the book first."
            
            self.progress.total_chunks = len(chunks)
            
            # Configure generation quality
            generation_kwargs = None
            if fast_preview:
                # Reduced quality for faster CPU generation
                generation_kwargs = {
                    'max_mel_tokens': 1200,  # Reduce from default 1500 but keep high enough
                    'num_beams': 2,  # Reduce from default 3
                    'temperature': 0.9,  # Slightly higher for faster sampling
                    'diffusion_steps': 15,  # Reduce from default 25
                    'inference_cfg_rate': 0.5,  # Reduce from default 0.7
                }
                print("⚡ Fast Preview Mode: Using reduced quality settings for faster generation")
                print("   ⚠️  Note: Final audiobook should be regenerated without Fast Preview for best quality")
            else:
                print("✨ High Quality Mode: Using default IndexTTS settings (slower but better quality)")
            
            # Generate audio for each chunk with terminal progress bar
            audio_files = []
            
            print(f"\n🎬 Starting audiobook generation ({len(chunks)} chunks)...")
            chunk_progress = tqdm(
                chunks, 
                desc="🎵 Generating audio",
                unit="chunk",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
            
            for i, chunk in enumerate(chunk_progress):
                self.progress.processed_chunks = i
                self.progress.current_task = f"Generating audio for chunk {i+1}/{len(chunks)}"
                
                # Update terminal progress description
                chunk_progress.set_description(f"🎵 Chunk {i+1}/{len(chunks)}")
                
                # Determine speaker for this chunk and resolve voice according to UI mapping
                speaker = chunk.get('speaker', 'Narrator')
                voice = self._get_voice_for_speaker(speaker, voice_assignments)
                
                # Debug output
                print(f"\n📝 Processing chunk {i+1}:")
                print(f"   Speaker: {speaker}")
                print(f"   Voice: {voice}")
                print(f"   Text length: {len(chunk['text'])} chars")
                
                if not voice or voice == "(Not Assigned)":
                    print(f"   ⚠️  WARNING: No voice assigned for {speaker}, skipping...")
                    continue
                
                # Generate audio
                print(f"   🎙️  Generating audio with {voice}...")
                audio_file = self.voice_processor.generate_chunk_audio(
                    text=chunk['text'],
                    voice_name=voice,
                    chunk_id=i,
                    book_name=book_file,
                    generation_kwargs=generation_kwargs,
                    emotion_type=emotion_type,
                    emotion_alpha=emotion_alpha
                )
                
                if audio_file:
                    print(f"   ✅ Generated: {audio_file}")
                    audio_files.append(audio_file)
                else:
                    print(f"   ❌ Failed to generate audio for chunk {i+1}")

            
            # Close the progress bar
            chunk_progress.close()
            print(f"✅ Audio generation complete! Generated {len(audio_files)} audio files.")
            
            self.progress.current_task = "Compiling final audiobook..."
            print("🔧 Compiling final audiobook...")
            
            # Compile all audio chunks into final audiobook using appropriate compiler
            audio_compiler = self.audio_compiler_single if is_single_speaker else self.audio_compiler_multi
            
            # Get voice name for single speaker mode
            mode = "single" if is_single_speaker else "multi"
            voice_name = None
            if is_single_speaker and voice_assignments:
                # Get the voice from assignments (usually Narrator in single speaker mode)
                voice_name = voice_assignments.get("Narrator") or list(voice_assignments.values())[0]
            
            final_audiobook = audio_compiler.compile_audiobook(
                audio_files, book_file, mode=mode, voice_name=voice_name
            )
            
            self.progress.status = "complete"
            self.progress.current_task = "Audiobook generation complete!"
            
            return f"Audiobook generated successfully: {final_audiobook}"
            
        except Exception as e:
            self.progress.error = str(e)
            self.progress.status = "error"
            return f"Error generating audiobook: {str(e)}"
    
    def get_progress_info(self) -> Tuple[str, int, str]:
        """Get current progress information with enhanced details"""
        if self.progress.total_chunks > 0:
            progress_percent = int((self.progress.processed_chunks / self.progress.total_chunks) * 100)
        else:
            progress_percent = 0
            
        # Status emoji mapping
        status_emojis = {
            "idle": "⏳",
            "analyzing": "🔍",
            "generating": "🎵",
            "complete": "✅",
            "error": "❌"
        }
        
        status_emoji = status_emojis.get(self.progress.status, "🔄")
        
        status_text = f"{status_emoji} Status: {self.progress.status.upper()}\n"
        status_text += f"📋 Current Task: {self.progress.current_task}\n"
        
        if self.progress.total_chunks > 0:
            status_text += f"📊 Progress: {self.progress.processed_chunks}/{self.progress.total_chunks} chunks"
            status_text += f" ({progress_percent}%)\n"
            
            if self.progress.status == "generating" and self.progress.processed_chunks > 0:
                # Estimate time remaining
                import time
                if hasattr(self.progress, 'start_time'):
                    elapsed = time.time() - self.progress.start_time
                    rate = self.progress.processed_chunks / elapsed if elapsed > 0 else 0
                    if rate > 0:
                        remaining_chunks = self.progress.total_chunks - self.progress.processed_chunks
                        eta_seconds = remaining_chunks / rate
                        eta_minutes = int(eta_seconds / 60)
                        eta_seconds = int(eta_seconds % 60)
                        status_text += f"⏱️ ETA: {eta_minutes}m {eta_seconds}s\n"
        
        if self.progress.error:
            status_text += f"\n❌ Error: {self.progress.error}"
            
        return status_text, progress_percent, self.progress.current_task

# Initialize the generator
generator = AudiobookGenerator()

def refresh_book_list():
    """Refresh the list of available books"""
    return gr.update(choices=generator.get_book_files())

def refresh_voice_samples():
    """Reload voice samples from directory"""
    generator.voice_processor.reload_voice_samples()
    voices = generator.voice_processor.get_available_voices()
    if voices:
        return f"{len(voices)} voice samples loaded"
    else:
        return "No voice samples found. Add .wav/.mp3 files to voice_samples folder"


def upload_voice_sample_ui(file):
    """Handler to save uploaded voice sample from Gradio UI"""
    if not file:
        return "❌ No file selected", gr.update(choices=generator.voice_processor.get_available_voices()), gr.update(choices=generator.voice_processor.get_available_voices()), gr.update(choices=generator.voice_processor.get_available_voices()), gr.update(choices=generator.voice_processor.get_available_voices()), ""
    try:
        # Gradio provides a temporary file path in file.name
        src_path = file.name
        original_name = Path(file.name).name
        success, msg = generator.voice_processor.save_voice_sample(src_path, original_filename=original_name)

        # Refresh available voices list
        voices = generator.voice_processor.get_available_voices()
        if voices:
            dropdown_update = gr.update(choices=voices, value=voices[0])
        else:
            dropdown_update = gr.update(choices=[], value=None)

        return (f"✅ {msg}" if success else f"❌ {msg}"), dropdown_update, dropdown_update, dropdown_update, dropdown_update, msg
    except Exception as e:
        return f"❌ Upload failed: {str(e)}", gr.update(), gr.update(), gr.update(), gr.update(), str(e)


def delete_voice_sample_ui(voice_name: str):
    """Handler to delete a voice sample selected in UI"""
    if not voice_name:
        return "❌ No voice selected to delete", gr.update(choices=generator.voice_processor.get_available_voices()), gr.update(choices=generator.voice_processor.get_available_voices()), gr.update(choices=generator.voice_processor.get_available_voices()), gr.update(choices=generator.voice_processor.get_available_voices()), ""
    try:
        success, msg = generator.voice_processor.delete_voice_sample(voice_name)
        voices = generator.voice_processor.get_available_voices()
        if voices:
            dropdown_update = gr.update(choices=voices, value=voices[0])
        else:
            dropdown_update = gr.update(choices=[], value=None)

        return (f"✅ {msg}" if success else f"❌ {msg}"), dropdown_update, dropdown_update, dropdown_update, dropdown_update, msg
    except Exception as e:
        return f"❌ Delete failed: {str(e)}", gr.update(), gr.update(), gr.update(), gr.update(), str(e)


def preview_voice_sample_ui(voice_name: str):
    """Handler to load a voice sample for preview playback"""
    if not voice_name:
        return None, "❌ No voice selected to preview"
    try:
        sample_path = generator.voice_processor.get_voice_sample_path(voice_name)
        if not sample_path:
            return None, f"❌ Voice sample '{voice_name}' not found"
        
        from pathlib import Path
        if not Path(sample_path).exists():
            return None, f"❌ Voice sample file not found: {sample_path}"
        
        return sample_path, f"✅ Loaded voice sample: {voice_name}"
    except Exception as e:
        return None, f"❌ Preview failed: {str(e)}"

def analyze_book(book_file: str, minor_to_narrator: bool = True, line_threshold: int = 5):
    """Analyze selected book for speakers"""
    if not book_file:
        return ("Please select a book file first.", {}, {}, gr.update(visible=False), [], "Please select a book first", 
                gr.update(visible=False), *[gr.update(visible=False) for _ in range(10)])
    
    book_path = Path("books") / book_file
    speaker_analysis, message = generator.analyze_book_speakers(str(book_path))
    
    if speaker_analysis:
        # Create speaker list for dataframe
        speaker_components = []
        voice_assignments = {}
        minor_characters = []
        
        speakers_list = list(speaker_analysis.items())
        
        for speaker, info in speaker_analysis.items():
            gender = info.get('gender', 'unknown')
            # Ensure lines is an integer - Gemini might return it as a string
            lines_val = info.get('line_count') or info.get('lines') or 0
            try:
                lines = int(lines_val) if lines_val else 0
            except (ValueError, TypeError):
                lines = 0
            char_type = info.get('type', 'unknown')
            description = info.get('description', 'No description')
            
            # Check if this is a minor character
            is_minor = minor_to_narrator and lines < line_threshold and speaker.lower() != "narrator"
            
            if is_minor:
                minor_characters.append(speaker)
                speaker_info = f"{speaker} ({gender}, {lines} lines, {char_type}) [MINOR]"
            else:
                speaker_info = f"{speaker} ({gender}, {lines} lines, {char_type})"
            
            # Don't auto-assign - let user choose or use auto-assign button
            voice_assignments[speaker] = "(Not Assigned)"
            
            speaker_components.append([speaker_info, description])
        
        # Create voice assignment status message
        voice_status_msg = f"✅ Found {len(speaker_analysis)} characters. Use Auto-Assign button or select voices manually:\n"
        
        if minor_characters:
            voice_status_msg += f"\n📢 Minor characters (< {line_threshold} lines):\n"
            for minor_char in minor_characters:
                lines = speaker_analysis[minor_char].get('lines', 0)
                voice_status_msg += f"• {minor_char} ({lines} lines)\n"
            voice_status_msg += "\n🎭 Main characters:\n"
        
        for speaker in speaker_analysis.keys():
            if speaker not in minor_characters:
                lines = speaker_analysis[speaker].get('lines', 0)
                voice_status_msg += f"• {speaker} ({lines} lines)\n"
        
        voice_status_msg += "\n\n👇 Select voices manually or click 'Auto-Assign Voices' button."
        
        # Get available custom voices
        available_voices = generator.voice_processor.get_available_voices()
        if not available_voices:
            available_voices = ["default"]
        
        # Add "(Not Assigned)" option at the beginning
        dropdown_choices = ["(Not Assigned)"] + available_voices
        
        # Create dropdown updates for each character
        dropdown_updates = []
        all_speakers = list(speaker_analysis.items())
        
        for i in range(10):  # Support up to 10 characters
            if i < len(all_speakers):
                speaker_name, speaker_info = all_speakers[i]
                gender = speaker_info.get('gender', 'unknown')
                lines = speaker_info.get('lines', 0)
                char_type = speaker_info.get('type', 'unknown')
                assigned_voice = voice_assignments.get(speaker_name, "(Not Assigned)")
                
                dropdown_updates.append(gr.update(
                    label=f"🎭 {speaker_name} ({gender}, {lines} lines)",
                    choices=dropdown_choices,  # Use dropdown_choices with "(Not Assigned)"
                    value=assigned_voice,
                    visible=True,
                    interactive=True
                ))
            else:
                dropdown_updates.append(gr.update(visible=False))
        
        return (
            message, 
            voice_assignments,  # Store voice assignments in JSON
            speaker_analysis,  # Store speaker analysis for auto-assign
            gr.update(visible=True),  # voice_assignment_section
            speaker_components,
            voice_status_msg,
            gr.update(visible=True),  # voice_dropdowns_container
            *dropdown_updates  # All 10 dropdown updates
        )
    else:
        return (
            message, 
            {},  # Empty voice assignments
            {},  # Empty speaker analysis
            gr.update(visible=False), 
            [], 
            "Analysis failed - no characters detected",
            gr.update(visible=False),  # voice_dropdowns_container
            *[gr.update(visible=False) for _ in range(10)]  # All 10 dropdowns hidden
        )

def create_voice_assignment_interface(speaker_data):
    """Create dynamic voice assignment interface"""
    if not speaker_data:
        return []
    
    components = []
    for speaker, info in speaker_data.items():
        gender = info.get('gender', 'unknown')
        suggested_voice = suggest_voice(gender)
        
        component = gr.Dropdown(
            choices=generator.available_voices,
            value=suggested_voice,
            label=f"{speaker} ({gender})",
            info=f"{info.get('lines', 0)} lines, {info.get('type', 'character')}"
        )
        components.append(component)
    
    return components

def suggest_voice(gender: str) -> str:
    """Suggest appropriate voice based on gender"""
    male_voices = ["leo", "dan", "zac"]
    female_voices = ["tara", "leah", "jess", "mia", "zoe"]
    
    if gender.lower() in ['male', 'm']:
        return male_voices[0]  # Default male voice
    elif gender.lower() in ['female', 'f']:
        return female_voices[0]  # Default female voice
    else:
        return "tara"  # Default narrator voice

def assign_voice_for_character(speaker_name, character_info):
    """Assign an appropriate voice for a character based on their info"""
    gender = character_info.get('gender', 'unknown').lower()
    char_type = character_info.get('type', 'unknown').lower()
    
    # Get available custom voices from voice processor
    available_voices = generator.voice_processor.get_available_voices()
    
    if not available_voices:
        return "default"
    
    # Categorize available voices by likely gender based on common names
    # This is a simple heuristic - users can always manually change assignments
    female_names = ['belle', 'nicole', 'ellen', 'nekomata', 'koleda', 'rina', 'qingyi', 'evelyn', 'vivian', 'burnice', 'astra', 'yuzuha', 'caesar', 'yixuan', 'zhuyuan']
    male_names = ['ben', 'billy', 'lycaon', 'lighter', 'wise', 'hugo', 'trigger', 'harumasa', 'jufufu', 'unagi']
    
    female_voices = [v for v in available_voices if any(name in v.lower() for name in female_names)]
    male_voices = [v for v in available_voices if any(name in v.lower() for name in male_names)]
    
    # Use a hash of the speaker name to deterministically pick a voice
    # This ensures same character gets same voice, but different characters get different voices
    import hashlib
    name_hash = int(hashlib.md5(speaker_name.encode()).hexdigest(), 16)
    
    # Special assignment for narrator - pick a distinct voice
    if speaker_name.lower() == "narrator":
        # Prefer male voices for narrator, or just pick one
        narrator_options = male_voices if male_voices else available_voices
        return narrator_options[name_hash % len(narrator_options)]
    
    # Assign based on gender with rotation
    if gender == "female" and female_voices:
        # Use hash to pick from female voices
        return female_voices[name_hash % len(female_voices)]
    elif gender == "male" and male_voices:
        # Use hash to pick from male voices
        return male_voices[name_hash % len(male_voices)]
    else:
        # Default: pick from all available voices using hash
        return available_voices[name_hash % len(available_voices)]

def create_voice_assignment_interface(speaker_analysis):
    """Create dynamic voice assignment dropdowns for each character"""
    if not speaker_analysis:
        return []
    
    # Get available voices from voice processor
    available_voices = generator.voice_processor.get_available_voices()
    if not available_voices:
        available_voices = ["default"]
    voice_dropdowns = []
    
    with gr.Column() as voice_column:
        gr.Markdown("### 🎭 Individual Voice Assignments")
        gr.Markdown("*Select a voice for each character below:*")
        
        for speaker_name, info in speaker_analysis.items():
            gender = info.get('gender', 'unknown')
            char_type = info.get('type', 'unknown')
            description = info.get('description', 'No description')
            
            # Create a more readable label
            gender_emoji = "👤" if gender == "unknown" else ("👨" if gender == "male" else "👩")
            label = f"{gender_emoji} {speaker_name} ({gender}, {char_type})"
            
            # Start with no voice selected - user can auto-assign or manually select
            # Add None option at the beginning for "not assigned"
            dropdown_choices = ["(Not Assigned)"] + available_voices
            
            dropdown = gr.Dropdown(
                choices=dropdown_choices,
                value="(Not Assigned)",  # Default to not assigned
                label=label,
                info=f"Description: {description} | Auto-assign button will suggest voices"
            )
            voice_dropdowns.append((speaker_name, dropdown))
    
    return voice_dropdowns

def auto_assign_voices(speaker_analysis_json, minor_to_narrator: bool = True, line_threshold: int = 5):
    """Auto-assign voices based on character analysis"""
    if not speaker_analysis_json:
        return "{}", "No character analysis available for auto-assignment."
    
    try:
        import json
        
        # Parse the speaker analysis
        if isinstance(speaker_analysis_json, str):
            try:
                speaker_analysis = json.loads(speaker_analysis_json)
            except:
                # If it's not valid JSON, treat it as empty
                return "{}", "Invalid speaker analysis data."
        else:
            speaker_analysis = speaker_analysis_json
        
        # Check if it's actually a dict (not a string from another field)
        if not isinstance(speaker_analysis, dict):
            return "{}", f"Speaker analysis is not a dictionary: {type(speaker_analysis)}"
        
        assignments = {}
        minor_characters = []
        
        for speaker_name, info in speaker_analysis.items():
            # Make sure info is a dict
            if not isinstance(info, dict):
                continue
                
            lines = info.get('lines', 0)
            
            # Check if this is a minor character
            is_minor = minor_to_narrator and lines < line_threshold and speaker_name.lower() != "narrator"
            
            if is_minor:
                # Assign to same voice as narrator
                narrator_voice = assignments.get("Narrator") if "Narrator" in speaker_analysis else None
                if not narrator_voice:
                    narrator_info = speaker_analysis.get("Narrator", {})
                    if isinstance(narrator_info, dict):
                        narrator_voice = assign_voice_for_character("Narrator", narrator_info)
                    else:
                        narrator_voice = "Lycaon - ZZZ"  # Default narrator voice
                assignments[speaker_name] = narrator_voice
                minor_characters.append(speaker_name)
            else:
                voice = assign_voice_for_character(speaker_name, info)
                assignments[speaker_name] = voice
        
        # Create assignment summary
        summary = "🤖 Auto-assigned voices:\n"
        
        if minor_characters:
            narrator_voice = assignments.get("Narrator", "narrator voice")
            summary += f"\n📢 Minor characters (< {line_threshold} lines) assigned to narrator:\n"
            for minor_char in minor_characters:
                if isinstance(speaker_analysis.get(minor_char), dict):
                    lines = speaker_analysis[minor_char].get('lines', 0)
                    summary += f"• {minor_char} ({lines} lines) → {narrator_voice} (narrator)\n"
            summary += "\n🎭 Main characters with individual voices:\n"
        
        for speaker, voice in assignments.items():
            if speaker not in minor_characters:
                if isinstance(speaker_analysis.get(speaker), dict):
                    lines = speaker_analysis[speaker].get('lines', 0)
                    summary += f"• {speaker} ({lines} lines) → {voice}\n"
        
        # Return as JSON string
        return json.dumps(assignments), summary
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Auto-assign error:\n{error_details}")
        return "{}", f"Error in auto-assignment: {str(e)}"

def auto_assign_with_dropdown_updates(speaker_analysis_json, minor_to_narrator: bool, line_threshold: int):
    """Auto-assign voices and return dropdown updates"""
    # Call the regular auto_assign function
    assignments_json, summary = auto_assign_voices(speaker_analysis_json, minor_to_narrator, line_threshold)
    
    # Parse assignments to update dropdowns
    try:
        import json
        assignments = json.loads(assignments_json) if isinstance(assignments_json, str) else assignments_json
        speaker_names = list(assignments.keys())
        
        # Create dropdown updates for each character
        dropdown_updates = []
        for i in range(10):  # Support up to 10 characters
            if i < len(speaker_names):
                assigned_voice = assignments[speaker_names[i]]
                dropdown_updates.append(gr.update(value=assigned_voice))
            else:
                dropdown_updates.append(gr.update())
        
        return assignments_json, summary, *dropdown_updates
        
    except Exception as e:
        print(f"Error in auto_assign_with_dropdown_updates: {e}")
        # Return empty updates on error
        return assignments_json, summary, *[gr.update() for _ in range(10)]

def update_voice_assignments(voice_assignments_json, *dropdown_values):
    """Update voice assignments from dropdown selections"""
    if not voice_assignments_json:
        return "{}", "No assignments to update"
    
    try:
        import json
        if isinstance(voice_assignments_json, str):
            assignments = json.loads(voice_assignments_json)
        else:
            assignments = voice_assignments_json
        
        # Get speaker names in order (same order as dropdowns were created)
        speaker_names = list(assignments.keys())
        
        # Update assignments with dropdown values (only update non-empty values)
        for i, dropdown_value in enumerate(dropdown_values):
            if i < len(speaker_names) and dropdown_value and dropdown_value != "(Not Assigned)":
                assignments[speaker_names[i]] = dropdown_value
        
        # Create status message
        status_msg = "✅ Voice assignments updated:\n"
        assigned_count = 0
        for speaker, voice in assignments.items():
            if voice != "(Not Assigned)":
                status_msg += f"• {speaker} → {voice}\n"
                assigned_count += 1
        
        if assigned_count == 0:
            status_msg = "⚠️ No voices assigned yet. Select voices from dropdowns above."
        
        return json.dumps(assignments), status_msg
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Update assignments error:\n{error_details}")
        return voice_assignments_json, f"Error updating assignments: {str(e)}"
        return voice_assignments_json, f"Error updating assignments: {str(e)}"
    # Implementation depends on dynamic interface setup
    pass

def update_voice_assignment(voice_assignments_json, *dropdown_values):
    """Update voice assignments when dropdowns change"""
    try:
        import json
        
        if not voice_assignments_json:
            return voice_assignments_json, "No assignments to update"
            
        # Parse current assignments
        if isinstance(voice_assignments_json, str):
            assignments = json.loads(voice_assignments_json)
        else:
            assignments = voice_assignments_json
            
        # Get the character names from assignments (in order)
        character_names = list(assignments.keys())
        
        # Update assignments with new dropdown values
        updated_assignments = {}
        for i, char_name in enumerate(character_names):
            if i < len(dropdown_values) and dropdown_values[i] is not None:
                updated_assignments[char_name] = dropdown_values[i]
            else:
                updated_assignments[char_name] = assignments.get(char_name, "tara")
        
        # Create status message
        status_msg = "🎭 Custom voice assignments:\n"
        for char, voice in updated_assignments.items():
            status_msg += f"• {char} → {voice}\n"
        
        return updated_assignments, status_msg
        
    except Exception as e:
        return voice_assignments_json, f"Error updating assignments: {str(e)}"

def start_generation(book_file: str, fast_preview: bool, voice_assignments_json, emotion_type: str = "neutral", emotion_intensity: float = 1.0):
    """Start audiobook generation with voice assignments"""
    if not book_file:
        return "❌ Please select and analyze a book first."
    
    # Convert voice assignments from JSON
    try:
        import json
        if isinstance(voice_assignments_json, str):
            voice_dict = json.loads(voice_assignments_json)
        else:
            voice_dict = voice_assignments_json
    except Exception as e:
        return f"❌ Error parsing voice assignments: {str(e)}"
    
    if not voice_dict:
        return "❌ No voice assignments found. Please analyze the book and assign voices first."
    
    # Check for unassigned voices
    unassigned = [name for name, voice in voice_dict.items() if not voice or voice == "(Not Assigned)"]
    if unassigned:
        return f"❌ Please assign voices to all characters. Unassigned: {', '.join(unassigned)}"
    
    preview_msg = "⚡ Fast Preview Mode" if fast_preview else "✨ High Quality Mode"
    emotion_msg = f" | Emotion: {emotion_type} @ {emotion_intensity:.1f}" if emotion_type != "neutral" else ""
    
    # Start generation directly (not in background thread to see errors)
    try:
        result = generator.generate_audiobook(
            book_file, 
            voice_dict, 
            fast_preview=fast_preview,
            emotion_type=emotion_type,
            emotion_alpha=emotion_intensity
        )
        return f"✅ {result}{emotion_msg}"
    except Exception as e:
        return f"❌ Generation failed: {str(e)}"

def get_progress():
    """Get current generation progress"""
    # The UI expects two sets of progress outputs (multi-speaker and single-speaker)
    # generator.get_progress_info() returns (status_text, percent, current_task)
    status_text, percent, current_task = generator.get_progress_info()

    # For now, mirror the same progress into the single-speaker widgets so the
    # Timer can update both tabs. If in future single-speaker progress diverges,
    # implement a separate tracker in the generator and return its values here.
    return status_text, percent, current_task, status_text, percent, current_task

def analyze_book_simple(book_file: str):
    """Simple book analysis for single speaker mode - just extract text and create chunks"""
    if not book_file:
        return "❌ Please select a book first."
    
    try:
        print(f"📖 Analyzing book for single speaker mode: {book_file}")
        
        # Prepend books directory if not already included
        if not book_file.startswith(f"{BOOKS_DIR}/"):
            book_file = f"{BOOKS_DIR}/{book_file}"
        
        # Extract text from book (this will be cached)
        full_text = generator.book_analyzer.extract_text(book_file)
        print(f"📝 Extracted {len(full_text):,} characters from book")
        
        # Create a simple speaker analysis with just a narrator
        simple_speaker_analysis = {
            "Narrator": {
                "name": "Narrator",
                "description": "Single voice narrator",
                "line_count": 1,
                "sample_quotes": []
            }
        }
        
        # Create chunks for processing and cache them - use simple_mode=True to skip Gemini
        chunks = generator.book_analyzer.chunk_text(full_text, simple_speaker_analysis, simple_mode=True)
        generator.progress.total_chunks = len(chunks)
        print(f"📄 Created {len(chunks)} text chunks for processing")
        
        # Store the analysis and chunks in cache so generate_audiobook can find them
        # Must use the same cache keys that get_cached_analysis and get_cached_chunks expect
        import json
        cache_key_analysis = generator.book_analyzer._get_cache_key(full_text)
        
        # Save speaker analysis to cache with key "speakers" (not "speaker_analysis")
        generator.book_analyzer._save_to_cache(cache_key_analysis, "speakers", simple_speaker_analysis)
        
        # Save chunks to cache with combined key (text + speaker analysis)
        cache_key_chunks = generator.book_analyzer._get_cache_key(
            full_text + json.dumps(simple_speaker_analysis, sort_keys=True)
        )
        generator.book_analyzer._save_to_cache(cache_key_chunks, "chunks", chunks)
        
        return f"✅ Book analyzed successfully! Ready to generate with single voice.\n📄 {len(chunks)} chunks created from {len(full_text):,} characters."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Analysis failed: {str(e)}"

def start_single_speaker_generation(book_file: str, voice_name: str, fast_preview: bool, emotion_type: str = "neutral", emotion_intensity: float = 1.0):
    """Start single speaker audiobook generation"""
    if not book_file:
        return "❌ Please select a book first."
    
    if not voice_name or voice_name == "(Not Assigned)":
        return "❌ Please select a voice for the audiobook."
    
    preview_msg = "⚡ Fast Preview Mode" if fast_preview else "✨ High Quality Mode"
    emotion_msg = f" | Emotion: {emotion_type} @ {emotion_intensity:.1f}" if emotion_type != "neutral" else ""
    
    try:
        # Normalize book_file to just the filename (remove books/ prefix if present)
        if book_file.startswith(f"{BOOKS_DIR}/"):
            book_file = book_file[len(f"{BOOKS_DIR}/"):]
        
        # For single speaker, we'll use a simple approach: read the book and generate with one voice
        # We can reuse the existing generation logic but with a single voice assignment
        voice_dict = {"Narrator": voice_name}  # Assign everything to narrator with chosen voice
        
        result = generator.generate_audiobook(
            book_file, 
            voice_dict, 
            fast_preview=fast_preview, 
            is_single_speaker=True,
            emotion_type=emotion_type,
            emotion_alpha=emotion_intensity
        )
        return f"✅ {result} ({preview_msg}{emotion_msg})"
    except Exception as e:
        return f"❌ Generation failed: {str(e)}"

def clear_cache():
    """Clear all cached data (analysis and chunks)"""
    try:
        cache_dir = Path("cache")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(exist_ok=True)
            return "✅ Cache cleared successfully! All cached book analyses have been removed."
        else:
            return "ℹ️ Cache directory does not exist. Nothing to clear."
    except Exception as e:
        return f"❌ Failed to clear cache: {str(e)}"

def test_voice_generation(voice_name: str, test_text: str, emotion_type: str = "neutral", emotion_intensity: float = 1.0):
    """Generate a test audio clip with the selected voice"""
    if not voice_name or voice_name == "(Not Assigned)":
        return "❌ Please select a voice to test.", None, None
    
    if not test_text or not test_text.strip():
        return "❌ Please enter some text to generate.", None, None
    
    try:
        # Create output directory for test audio
        test_output_dir = Path("output/test_audio")
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename: "Voice Name - Emotion @ Intensity.wav"
        # Example: "Astra - ZZZ - calm @ 0.7.wav" or "Astra - ZZZ - neutral.wav"
        safe_voice_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '-', '_')).strip()
        
        if emotion_type != "neutral" and emotion_intensity > 0.0:
            # Include emotion and intensity
            filename = f"{safe_voice_name} - {emotion_type} @ {emotion_intensity:.1f}.wav"
        else:
            # Just voice name (neutral)
            filename = f"{safe_voice_name} - neutral.wav"
        
        output_file = test_output_dir / filename
        
        # Generate a timestamp-based chunk ID for uniqueness
        import time
        timestamp = int(time.time())
        
        # Generate the audio using the voice processor's generate_chunk_audio method
        emotion_msg = f" with {emotion_type} emotion @ {emotion_intensity:.1f}" if emotion_type != "neutral" else ""
        print(f"🎵 Generating test audio with voice: {voice_name}{emotion_msg}")
        
        # Use a temp book name for chunking, then rename
        temp_audio_path = generator.voice_processor.generate_chunk_audio(
            text=test_text.strip(),
            voice_name=voice_name,
            chunk_id=timestamp,  # Use timestamp as unique chunk ID
            book_name="test_audio",
            emotion_type=emotion_type,
            emotion_alpha=emotion_intensity
        )
        
        if temp_audio_path and Path(temp_audio_path).exists():
            # Rename to our nice format
            import shutil
            shutil.move(temp_audio_path, output_file)
            print(f"✅ Generated audio saved to: {output_file.name}")
            
            # Return path for both audio player and file download
            return f"✅ Test audio generated successfully with {voice_name}{emotion_msg}!", str(output_file), str(output_file)
        else:
            return "❌ Failed to generate audio - check logs for details.", None, None
    
    except Exception as e:
        import traceback
        print(f"❌ Error generating test audio: {str(e)}")
        print(traceback.format_exc())
        return f"❌ Failed to generate test audio: {str(e)}", None, None

def upload_book(file, subfolder=""):
    """Upload a book file to the books directory (with optional subfolder)"""
    if file is None:
        return "❌ No file selected", gr.Dropdown(choices=generator.get_book_files())
    
    try:
        import shutil
        books_dir = Path(BOOKS_DIR)
        books_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the original filename
        filename = Path(file.name).name
        
        # Handle subfolder if provided
        if subfolder and subfolder.strip():
            # Clean up subfolder name (remove leading/trailing slashes and spaces)
            subfolder_clean = subfolder.strip().strip('/')
            target_dir = books_dir / subfolder_clean
            target_dir.mkdir(parents=True, exist_ok=True)
            destination = target_dir / filename
            relative_path = f"{subfolder_clean}/{filename}"
        else:
            destination = books_dir / filename
            relative_path = filename
        
        # Check if file already exists
        if destination.exists():
            return f"⚠️ Book '{relative_path}' already exists. Please rename or delete the existing file first.", gr.Dropdown(choices=generator.get_book_files())
        
        # Copy the uploaded file to books directory
        shutil.copy2(file.name, destination)
        
        # Refresh the book list
        updated_choices = generator.get_book_files()
        
        return f"✅ Book '{relative_path}' uploaded successfully!", gr.Dropdown(choices=updated_choices, value=relative_path)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Upload failed: {str(e)}", gr.Dropdown(choices=generator.get_book_files())

def delete_book(book_file):
    """Delete a book file from the books directory (handles subfolders)"""
    if not book_file:
        return "❌ No book selected", gr.Dropdown(choices=generator.get_book_files())
    
    try:
        books_dir = Path(BOOKS_DIR)
        book_path = books_dir / book_file
        
        if not book_path.exists():
            return f"❌ Book '{book_file}' not found", gr.Dropdown(choices=generator.get_book_files())
        
        # Delete the file
        book_path.unlink()
        
        # Check if parent directory is empty and remove it (but not the root books folder)
        parent = book_path.parent
        if parent != books_dir and not any(parent.iterdir()):
            parent.rmdir()
            print(f"🗑️ Removed empty subfolder: {parent.relative_to(books_dir)}")
        
        # Also clear any cached data for this book
        try:
            from book_analyzer import BookAnalyzer
            analyzer = BookAnalyzer()
            # Need to construct the full path for cache clearing
            full_path = f"{BOOKS_DIR}/{book_file}"
            text = analyzer.extract_text(full_path)
            cache_key = analyzer._get_cache_key(text)
            
            # Remove cache files if they exist
            cache_dir = Path("cache")
            if cache_dir.exists():
                for cache_file in cache_dir.glob(f"*{cache_key}*"):
                    cache_file.unlink()
        except:
            pass  # Ignore cache clearing errors
        
        # Refresh the book list
        updated_choices = generator.get_book_files()
        
        return f"✅ Book '{book_file}' deleted successfully!", gr.Dropdown(choices=updated_choices, value=updated_choices[0] if updated_choices else None)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Delete failed: {str(e)}", gr.Dropdown(choices=generator.get_book_files())

def refresh_single_book_list():
    """Refresh book list for single speaker tab"""
    return gr.Dropdown(choices=generator.get_book_files())

# Create Gradio interface
with gr.Blocks(title="AI Audiobook Generator", theme=gr.themes.Soft()) as app:
    gr.HTML("""
    <h1 style="text-align: center; color: #2563eb;">🎧 AI Audiobook Generator</h1>
    <p style="text-align: center;">Generate professional audiobooks with multiple AI voices using IndexTTS one-shot cloning</p>
    """)
    
    # Voice Samples Management Section
    with gr.Accordion("🎤 Voice Samples Management", open=False):
        gr.HTML("""
            <p><strong>ℹ️ About Voice Samples:</strong> Add your own voice samples to the <code>voice_samples</code> folder for custom characters!</p>
            <p>• Each audio file should be 5-15 seconds of clear speech (up to 60s supported)<br/>
            • Name files after your characters (e.g., <code>Narrator.wav</code>, <code>Hero.wav</code>)<br/>
            • Supported formats: WAV, MP3, FLAC, OGG<br/>
            • Current voices will appear in dropdowns when assigning characters after book analysis</p>
        """)
        
        with gr.Row():
            voices_count = gr.Textbox(
                label="Voice Samples Status",
                value=f"{len(generator.voice_processor.get_available_voices())} voice samples loaded" if generator.voice_processor.get_available_voices() else "No voice samples found. Add .wav/.mp3 files to voice_samples folder",
                interactive=False,
                lines=1
            )
            with gr.Column(scale=0, min_width=150):
                refresh_voices_btn = gr.Button("🔄 Reload Voices", size="sm")

        # Upload/Delete voice sample controls
        with gr.Row():
            voice_upload_file = gr.File(
                label="📤 Upload Voice Sample",
                file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                file_count="single"
            )
            voice_upload_status = gr.Textbox(label="Upload Status", interactive=False)

        with gr.Row():
            voice_delete_dropdown = gr.Dropdown(
                choices=generator.voice_processor.get_available_voices(),
                label="Select Voice to Delete",
                value=None
            )
            with gr.Column(scale=0, min_width=150):
                voice_delete_btn = gr.Button("🗑️ Delete Voice", variant="stop", size="sm")
        
        # Voice sample preview section
        gr.HTML("<h3>🔊 Voice Sample Preview</h3>")
        with gr.Row():
            voice_preview_dropdown = gr.Dropdown(
                choices=generator.voice_processor.get_available_voices(),
                label="Select Voice to Preview",
                value=None
            )
            with gr.Column(scale=0, min_width=150):
                voice_preview_btn = gr.Button("▶️ Load Preview", variant="primary", size="sm")
        
        voice_preview_audio = gr.Audio(
            label="🎵 Voice Sample Player",
            type="filepath",
            interactive=False
        )
        voice_preview_status = gr.Textbox(label="Preview Status", interactive=False)
    
    # Create tabs for different generation modes
    with gr.Tabs() as main_tabs:
        # Tab 1: Multi Speaker Generation
        with gr.TabItem("🎭 Multi Speaker Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("<h2>📚 Book Selection</h2>")
                    
                    with gr.Row():
                        book_dropdown = gr.Dropdown(
                            choices=generator.get_book_files(),
                            label="Select Book",
                            info=f"Choose from books in '{BOOKS_DIR}' folder (scans subfolders)"
                        )
                        refresh_btn = gr.Button("🔄 Refresh", scale=0)
                    
                    # Upload and Delete section
                    with gr.Row():
                        upload_file = gr.File(
                            label="📤 Upload Book",
                            file_types=[".txt", ".pdf", ".epub", ".docx", ".odt", ".rtf", ".doc"],
                            file_count="single"
                        )
                        with gr.Column(scale=0, min_width=100):
                            delete_btn = gr.Button("🗑️ Delete", variant="stop", size="sm")
                    
                    with gr.Row():
                        upload_subfolder = gr.Textbox(
                            label="📁 Subfolder (optional)",
                            placeholder="e.g., Stephen King, Tolkien, etc.",
                            info="Leave empty to upload to root books folder"
                        )
                    
                    upload_status = gr.Textbox(label="Upload/Delete Status", interactive=False, visible=False)
                    
                    upload_info = gr.HTML("""
                        <p><strong>Supported formats:</strong> TXT, PDF, EPUB, DOCX, ODT, RTF, DOC</p>
                        <p><strong>💡 Tip:</strong> Books are scanned from all subfolders automatically!</p>
                    """)
                    
                    analyze_btn = gr.Button("🔍 Analyze Book", variant="primary")
                    analysis_status = gr.Textbox(label="Analysis Status", interactive=False)
                
                with gr.Column(scale=1):
                    gr.HTML("<h2>📊 System Status</h2>")
                    progress_text = gr.Textbox(label="Progress", interactive=False)
                    progress_bar = gr.Slider(
                        minimum=0, 
                        maximum=100, 
                        value=0, 
                        label="Progress (%)",
                        interactive=False
                    )
                    current_task = gr.Textbox(label="Current Task", interactive=False)
            
            with gr.Row(visible=False) as voice_assignment_section:
                with gr.Column():
                    gr.HTML("<h2>🎭 Voice Assignments</h2>")
                    speaker_list = gr.Dataframe(
                        headers=["Speaker Info", "Description"],
                        label="Detected Characters",
                        interactive=False
                    )
                    
                    gr.HTML("<h3>🎤 Voice Assignments</h3>")
                    
                    # Minor character handling options
                    with gr.Row():
                        with gr.Column(scale=1):
                            minor_to_narrator = gr.Checkbox(
                                label="📢 Assign minor characters to narrator",
                                value=True,
                                info="Characters with few lines will use narrator voice"
                            )
                        with gr.Column(scale=1):
                            line_threshold = gr.Number(
                                label="🔢 Minimum lines for own voice",
                                value=5,
                                minimum=1,
                                maximum=50,
                                step=1,
                                info="Characters with fewer lines will use narrator voice"
                            )
                    
                    # Auto-assign button and status
                    with gr.Row():
                        auto_assign_btn = gr.Button("🤖 Auto-Assign Voices", variant="secondary")
                        assignment_status = gr.Textbox(
                            label="Assignment Status", 
                            value="Analyze a book first to see characters",
                            interactive=False,
                            max_lines=5
                        )
                    
                    # Hidden JSON to store voice assignments and speaker analysis
                    voice_assignments_json = gr.JSON(label="Voice Assignments", visible=False)
                    speaker_analysis_json = gr.JSON(label="Speaker Analysis", visible=False)
                    
                    # Dynamic voice assignment dropdowns (will be populated after analysis)
                    gr.Markdown("### 🎤 Assign Voices to Characters")
                    gr.Markdown("*Select a voice sample for each detected character:*")
                    
                    # Create container for dynamic dropdowns
                    voice_dropdowns_container = gr.Column(visible=False)
                    with voice_dropdowns_container:
                        # These will be dynamically created based on detected characters
                        character_voice_inputs = []
                        for i in range(10):  # Support up to 10 characters
                            dropdown = gr.Dropdown(
                                label=f"Character {i+1}",
                                choices=["(Not Assigned)"],  # Start with Not Assigned option
                                value="(Not Assigned)",
                                visible=False,
                                interactive=True,
                                allow_custom_value=False  # Strict validation
                            )
                            character_voice_inputs.append(dropdown)
                    
                    # Emotion control settings
                    gr.HTML("<h3>🎭 Emotion Control</h3>")
                    with gr.Row():
                        emotion_type = gr.Dropdown(
                            label="Emotion Type",
                            choices=["neutral", "calm", "happy", "sad", "angry"],
                            value="neutral",
                            info="Apply consistent emotion across all voices (calm is great for normalizing)"
                        )
                        emotion_intensity = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.1,
                            label="Emotion Intensity",
                            info="0.0 = none, 1.0 = full emotion"
                        )
                    
                    gr.HTML("""
                        <p style="font-size: 0.9em; color: #666;">
                        <strong>💡 Emotion Tips:</strong> <strong>Calm</strong> is excellent for normalizing voice samples and reducing over-expression. 
                        Try 0.5-0.7 intensity for subtle emotion control.
                        </p>
                    """)
                    
                    # Quality settings
                    with gr.Row():
                        fast_preview_mode = gr.Checkbox(
                            label="⚡ Fast Preview Mode (CPU Only)",
                            value=False,
                            info="Use reduced quality for 2-3x faster generation on CPU. ⚠️ Lower quality - use only for testing!"
                        )
                    
                    gr.HTML("""
                        <p style="font-size: 0.9em; color: #666;">
                        <strong>💡 Tip for CPU users:</strong> Fast Preview Mode reduces quality but speeds up generation significantly on CPU-only systems. 
                        For final production audiobooks, disable Fast Preview or deploy to GPU server via Docker.
                        </p>
                    """)
                    
                    # Generate button and Clear Cache button
                    with gr.Row():
                        generate_btn = gr.Button("🎬 Generate Audiobook", variant="primary", size="lg", scale=3)
                        clear_cache_btn_multi = gr.Button("🗑️ Clear Cache", variant="secondary", size="lg", scale=1)
                    
                    cache_status_multi = gr.Textbox(label="Cache Status", interactive=False, visible=False)
            
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h2>📁 Output Files</h2>")
                    output_files = gr.File(label="Generated Audiobooks", file_count="multiple")
                    generation_log = gr.Textbox(
                        label="Generation Log",
                        lines=5,
                        interactive=False
                    )
        
        # Tab 2: Single Speaker Generation
        with gr.TabItem("🎤 Single Speaker Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("<h2>📚 Book Selection & Analysis</h2>")
                    
                    with gr.Row():
                        single_book_dropdown = gr.Dropdown(
                            choices=generator.get_book_files(),
                            label="Select Book",
                            info=f"Choose from books in '{BOOKS_DIR}' folder (scans subfolders)"
                        )
                        single_refresh_btn = gr.Button("🔄 Refresh", scale=0)
                    
                    # Upload and Delete section
                    with gr.Row():
                        single_upload_file = gr.File(
                            label="📤 Upload Book",
                            file_types=[".txt", ".pdf", ".epub", ".docx", ".odt", ".rtf", ".doc"],
                            file_count="single"
                        )
                        with gr.Column(scale=0, min_width=100):
                            single_delete_btn = gr.Button("🗑️ Delete", variant="stop", size="sm")
                    
                    with gr.Row():
                        single_upload_subfolder = gr.Textbox(
                            label="📁 Subfolder (optional)",
                            placeholder="e.g., Stephen King, Tolkien, etc.",
                            info="Leave empty to upload to root books folder"
                        )
                    
                    single_upload_status = gr.Textbox(label="Upload/Delete Status", interactive=False, visible=False)
                    
                    upload_info_single = gr.HTML("""
                        <p><strong>Supported formats:</strong> TXT, PDF, EPUB, DOCX, ODT, RTF, DOC</p>
                        <p><strong>💡 Tip:</strong> Books are scanned from all subfolders automatically!</p>
                    """)
                    
                    single_analyze_btn = gr.Button("🔍 Analyze Book", variant="primary")
                    single_analysis_status = gr.Textbox(label="Analysis Status", interactive=False)
                    
                    gr.HTML("<h3>🎤 Assign Voice</h3>")
                    single_voice_dropdown = gr.Dropdown(
                        choices=["(Not Assigned)"] + generator.voice_processor.get_available_voices(),
                        label="Select Voice for Entire Book",
                        info="Choose a voice sample for the entire audiobook",
                        value="(Not Assigned)"
                    )
                    
                    # Emotion control settings
                    gr.HTML("<h3>🎭 Emotion Control</h3>")
                    with gr.Row():
                        single_emotion_type = gr.Dropdown(
                            label="Emotion Type",
                            choices=["neutral", "calm", "happy", "sad", "angry"],
                            value="neutral",
                            info="Apply consistent emotion (calm is great for normalizing)"
                        )
                        single_emotion_intensity = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.1,
                            label="Emotion Intensity",
                            info="0.0 = none, 1.0 = full emotion"
                        )
                    
                    gr.HTML("""
                        <p style="font-size: 0.9em; color: #666;">
                        <strong>💡 Emotion Tips:</strong> <strong>Calm</strong> is excellent for normalizing voice samples and reducing over-expression. 
                        Try 0.5-0.7 intensity for subtle emotion control.
                        </p>
                    """)
                    
                    # Quality settings
                    single_fast_preview_mode = gr.Checkbox(
                        label="⚡ Fast Preview Mode (CPU Only)",
                        value=False,
                        info="Use reduced quality for 2-3x faster generation on CPU. ⚠️ Lower quality - use only for testing!"
                    )
                    
                    # Generate button and Clear Cache button
                    with gr.Row():
                        single_generate_btn = gr.Button("🎬 Generate Single Speaker Audiobook", variant="primary", size="lg", scale=3)
                        clear_cache_btn_single = gr.Button("🗑️ Clear Cache", variant="secondary", size="lg", scale=1)
                    
                    cache_status_single = gr.Textbox(label="Cache Status", interactive=False, visible=False)
                    single_generation_status = gr.Textbox(label="Generation Status", interactive=False, lines=3)
                
                with gr.Column(scale=1):
                    gr.HTML("<h2>📊 System Status</h2>")
                    single_progress_text = gr.Textbox(label="Progress", interactive=False)
                    single_progress_bar = gr.Slider(
                        minimum=0, 
                        maximum=100, 
                        value=0, 
                        label="Progress (%)",
                        interactive=False
                    )
                    single_current_task = gr.Textbox(label="Current Task", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h2>📁 Output Files</h2>")
                    single_output_files = gr.File(label="Generated Audiobooks", file_count="multiple")
                    single_generation_log = gr.Textbox(
                        label="Generation Log",
                        lines=5,
                        interactive=False
                    )
        
        # Tab 3: Test Voice
        with gr.TabItem("🔊 Test Voice"):
            gr.HTML("<h2>🎙️ Voice Testing</h2>")
            gr.HTML("<p>Test how different voices sound before generating your audiobook</p>")
            
            with gr.Row():
                with gr.Column():
                    test_voice_dropdown = gr.Dropdown(
                        choices=generator.voice_processor.get_available_voices(),
                        label="Select Voice to Test",
                        info="Choose a voice sample to test"
                    )
                    
                    test_text_input = gr.Textbox(
                        label="Test Text",
                        placeholder="Enter a paragraph or sentence to hear how this voice sounds...",
                        lines=5,
                        value="The quick brown fox jumps over the lazy dog. This is a sample sentence to test the voice quality and characteristics."
                    )
                    
                    # Emotion control settings
                    gr.HTML("<h3>🎭 Emotion Control</h3>")
                    with gr.Row():
                        test_emotion_type = gr.Dropdown(
                            label="Emotion Type",
                            choices=["neutral", "calm", "happy", "sad", "angry"],
                            value="neutral",
                            info="Test different emotions with this voice"
                        )
                        test_emotion_intensity = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.1,
                            label="Emotion Intensity",
                            info="0.0 = none, 1.0 = full emotion"
                        )
                    
                    gr.HTML("""
                        <p style="font-size: 0.9em; color: #666;">
                        <strong>💡 Quick Test:</strong> Try <strong>calm @ 0.7</strong> for normalized voices, or <strong>happy @ 0.5</strong> for subtle positivity.
                        </p>
                    """)
                    
                    test_generate_btn = gr.Button("🎵 Generate Test Audio", variant="primary", size="lg")
                    test_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    test_audio_output = gr.Audio(
                        label="🎵 Test Audio Player", 
                        type="filepath",
                        interactive=False
                    )
                    test_audio_file = gr.File(
                        label="📥 Download Test Audio",
                        interactive=False,
                        file_count="single"
                    )
                    gr.HTML("""
                        <p><strong>💡 Tips:</strong></p>
                        <ul>
                            <li>Try different text to hear voice characteristics</li>
                            <li>Test with dialogue and narration samples</li>
                            <li>Compare multiple voices before final selection</li>
                            <li>Use the download button to save and compare revisions</li>
                        </ul>
                    """)
    
    # Event handlers
    refresh_voices_btn.click(
        fn=refresh_voice_samples,
        outputs=[voices_count]
    )

    # Voice sample upload/delete handlers
    voice_upload_file.upload(
        fn=upload_voice_sample_ui,
        inputs=[voice_upload_file],
        outputs=[voices_count, voice_delete_dropdown, voice_preview_dropdown, test_voice_dropdown, single_voice_dropdown, voice_upload_status]
    )

    voice_delete_btn.click(
        fn=lambda voice_name: (
            # Show confirmation modal via gr.Warning (Gradio 4.x uses gr.Warning for user prompts)
            gr.Warning(f"⚠️ Are you sure you want to delete '{voice_name}'? This cannot be undone."),
            voice_name  # Pass through to actual delete
        )[1],  # Return the voice_name to next step
        inputs=[voice_delete_dropdown],
        outputs=None  # Confirmation only
    ).then(
        fn=delete_voice_sample_ui,
        inputs=[voice_delete_dropdown],
        outputs=[voices_count, voice_delete_dropdown, voice_preview_dropdown, test_voice_dropdown, single_voice_dropdown, voice_upload_status]
    )
    
    voice_preview_btn.click(
        fn=preview_voice_sample_ui,
        inputs=[voice_preview_dropdown],
        outputs=[voice_preview_audio, voice_preview_status]
    )
    
    refresh_btn.click(
        fn=refresh_book_list,
        outputs=[book_dropdown]
    )
    
    # Multi Speaker Upload/Delete handlers
    upload_file.upload(
        fn=upload_book,
        inputs=[upload_file, upload_subfolder],
        outputs=[upload_status, book_dropdown]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[upload_status]
    )
    
    delete_btn.click(
        fn=delete_book,
        inputs=[book_dropdown],
        outputs=[upload_status, book_dropdown]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[upload_status]
    )
    
    analyze_btn.click(
        fn=analyze_book,
        inputs=[book_dropdown, minor_to_narrator, line_threshold],
        outputs=[
            analysis_status, 
            voice_assignments_json,
            speaker_analysis_json,  # Add speaker analysis output
            voice_assignment_section, 
            speaker_list,
            assignment_status,
            voice_dropdowns_container,
            *character_voice_inputs
        ]
    )
    
    auto_assign_btn.click(
        fn=auto_assign_with_dropdown_updates,
        inputs=[speaker_analysis_json, minor_to_narrator, line_threshold],
        outputs=[voice_assignments_json, assignment_status, *character_voice_inputs]
    )
    
    # Add change events for all voice assignment dropdowns
    for dropdown in character_voice_inputs:
        dropdown.change(
            fn=update_voice_assignments,
            inputs=[voice_assignments_json, *character_voice_inputs],
            outputs=[voice_assignments_json, assignment_status]
        )
    
    generate_btn.click(
        fn=start_generation,
        inputs=[book_dropdown, fast_preview_mode, voice_assignments_json, emotion_type, emotion_intensity],
        outputs=[generation_log]
    )
    
    # Clear Cache button handler for Multi Speaker tab
    def show_cache_status(result):
        """Show the cache status message"""
        return gr.update(value=result, visible=True)
    
    clear_cache_btn_multi.click(
        fn=clear_cache,
        outputs=[cache_status_multi]
    ).then(
        fn=lambda x: gr.update(value=x, visible=True),
        inputs=[cache_status_multi],
        outputs=[cache_status_multi]
    )
    
    # Single Speaker Generation event handlers
    single_refresh_btn.click(
        fn=refresh_single_book_list,
        outputs=[single_book_dropdown]
    )
    
    # Single Speaker Upload/Delete handlers
    single_upload_file.upload(
        fn=upload_book,
        inputs=[single_upload_file, single_upload_subfolder],
        outputs=[single_upload_status, single_book_dropdown]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[single_upload_status]
    )
    
    single_delete_btn.click(
        fn=delete_book,
        inputs=[single_book_dropdown],
        outputs=[single_upload_status, single_book_dropdown]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[single_upload_status]
    )
    
    single_analyze_btn.click(
        fn=analyze_book_simple,
        inputs=[single_book_dropdown],
        outputs=[single_analysis_status]
    )
    
    single_generate_btn.click(
        fn=start_single_speaker_generation,
        inputs=[single_book_dropdown, single_voice_dropdown, single_fast_preview_mode, single_emotion_type, single_emotion_intensity],
        outputs=[single_generation_log]
    )
    
    # Clear Cache button handler for Single Speaker tab
    clear_cache_btn_single.click(
        fn=clear_cache,
        outputs=[cache_status_single]
    ).then(
        fn=lambda x: gr.update(value=x, visible=True),
        inputs=[cache_status_single],
        outputs=[cache_status_single]
    )
    
    # Test Voice event handlers
    test_generate_btn.click(
        fn=test_voice_generation,
        inputs=[test_voice_dropdown, test_text_input, test_emotion_type, test_emotion_intensity],
        outputs=[test_status, test_audio_output, test_audio_file]
    )
    
    # Auto-refresh progress every 2 seconds during generation using Timer
    # This will update progress for both multi-speaker and single-speaker tabs
    timer = gr.Timer(2)
    timer.tick(
        fn=get_progress,
        outputs=[progress_text, progress_bar, current_task, single_progress_text, single_progress_bar, single_current_task]
    )

if __name__ == "__main__":
    print("🚀 Starting AI Audiobook Generator...")
    print(f"📂 Books directory: {Path(BOOKS_DIR).absolute()}")
    print(f"   💡 Set BOOKS_DIR environment variable to change location")
    print("🔑 Set your GEMINI_API_KEY environment variable")
    
    # Get server settings from environment or use defaults
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    
    print("🌐 Web UI will be available at:")
    print(f"   • Local: http://localhost:{server_port}")
    print(f"   • Network: http://{server_name}:{server_port}")
    if server_name == "0.0.0.0":
        print(f"   • Access from other devices on your network using your server's IP")
    
    try:
        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=False,
            inbrowser=False  # Don't auto-open browser in Docker
        )
    except Exception as e:
        print(f"Failed to launch on {server_name}, trying 0.0.0.0...")
        print(f"If this works, use http://localhost:{server_port} in your browser")
        app.launch(
            server_name="0.0.0.0",  # Fallback to all interfaces
            server_port=server_port,
            share=False
        )