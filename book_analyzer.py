import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from google import genai
from pypdf import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import hashlib

# Document format support
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from odf import text as odf_text, teletype
    from odf.opendocument import load as odf_load
    ODT_AVAILABLE = True
except ImportError:
    ODT_AVAILABLE = False

try:
    from striprtf.striprtf import rtf_to_text
    RTF_AVAILABLE = True
except ImportError:
    RTF_AVAILABLE = False

try:
    import textract
    TEXTRACT_AVAILABLE = True
except ImportError:
    TEXTRACT_AVAILABLE = False

# Import API key from config file
try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = None
    print("Warning: config.py not found or GEMINI_API_KEY not set in config.py")

class BookAnalyzer:
    def __init__(self):
        # Configure Gemini / Google Gen AI
        # Try config file first, then environment variable as fallback
        api_key = GEMINI_API_KEY or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("API key required. Either set GEMINI_API_KEY in config.py or as environment variable")

        # Use the new google-genai client (imported as `from google import genai`)
        # The client will use the provided API key and expose `models.generate_content`.
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash')
        
        # Cache directory for analysis results
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def extract_text(self, book_path: str) -> str:
        """Extract text from various book formats"""
        book_path = Path(book_path)
        
        if not book_path.exists():
            raise FileNotFoundError(f"Book file not found: {book_path}")
        
        suffix = book_path.suffix.lower()
        
        if suffix == '.txt':
            return self._extract_from_txt(book_path)
        elif suffix == '.pdf':
            return self._extract_from_pdf(book_path)
        elif suffix == '.epub':
            return self._extract_from_epub(book_path)
        elif suffix == '.docx':
            return self._extract_from_docx(book_path)
        elif suffix == '.odt':
            return self._extract_from_odt(book_path)
        elif suffix == '.rtf':
            return self._extract_from_rtf(book_path)
        elif suffix == '.doc':
            return self._extract_from_doc(book_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .txt, .pdf, .epub, .docx, .odt, .rtf, .doc")
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def _extract_from_epub(self, file_path: Path) -> str:
        """Extract text from EPUB file"""
        try:
            book = epub.read_epub(str(file_path))
            text = ""
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + "\n"
            
            return text
        except Exception as e:
            raise Exception(f"Error reading EPUB: {str(e)}")
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")
        
        try:
            doc = DocxDocument(str(file_path))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def _extract_from_odt(self, file_path: Path) -> str:
        """Extract text from ODT file"""
        if not ODT_AVAILABLE:
            raise ImportError("odfpy not installed. Install with: pip install odfpy")
        
        try:
            doc = odf_load(str(file_path))
            all_paragraphs = doc.getElementsByType(odf_text.P)
            text = ""
            for paragraph in all_paragraphs:
                text += teletype.extractText(paragraph) + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading ODT: {str(e)}")
    
    def _extract_from_rtf(self, file_path: Path) -> str:
        """Extract text from RTF file"""
        if not RTF_AVAILABLE:
            raise ImportError("striprtf not installed. Install with: pip install striprtf")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                rtf_content = file.read()
            return rtf_to_text(rtf_content)
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                rtf_content = file.read()
            return rtf_to_text(rtf_content)
        except Exception as e:
            raise Exception(f"Error reading RTF: {str(e)}")
    
    def _extract_from_doc(self, file_path: Path) -> str:
        """Extract text from old DOC file using textract"""
        if not TEXTRACT_AVAILABLE:
            raise ImportError("textract not installed. Install with: pip install textract")
        
        try:
            text = textract.process(str(file_path)).decode('utf-8')
            return text
        except Exception as e:
            raise Exception(f"Error reading DOC: {str(e)}. Note: .doc support requires additional system dependencies.")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str, cache_type: str) -> Optional[Any]:
        """Load data from cache"""
        cache_file = self.cache_dir / f"{cache_key}_{cache_type}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, cache_type: str, data: Any):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}_{cache_type}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def analyze_speakers(self, text: str) -> Dict[str, Any]:
        """Analyze text to identify speakers using Gemini"""
        cache_key = self._get_cache_key(text)
        
        # Try to load from cache first
        cached_result = self._load_from_cache(cache_key, "speakers")
        if cached_result:
            return cached_result
        
        # Prepare text for analysis (truncate if too long)
        max_chars = 50000  # Gemini has token limits
        if len(text) > max_chars:
            # Take beginning, middle, and end samples
            sample_size = max_chars // 3
            text_sample = (
                text[:sample_size] + 
                text[len(text)//2 - sample_size//2:len(text)//2 + sample_size//2] + 
                text[-sample_size:]
            )
        else:
            text_sample = text
        
        prompt = f"""
        Analyze this book text and identify all characters who speak dialogue (text in quotes).
        
        IMPORTANT: Also include a "Narrator" entry - this represents ALL descriptive text, thoughts, actions, and scene descriptions that are NOT spoken dialogue.
        
        For example:
        - "The group walked down the dark path" = Narrator (descriptive text)
        - "This will be a fight" said Dan = Dan (dialogue)
        
        For each entry, provide:
        1. Character name (use "Narrator" for all non-dialogue narrative text)
        2. Gender (male/female/unknown)
        3. Approximate number of segments they speak/narrate
        4. Character type (main/supporting/minor/narrator)
        
        Return as JSON in this exact format:
        {{
            "Narrator": {{
                "gender": "neutral",
                "lines": <high number>,
                "type": "narrator",
                "description": "The narrative voice - all descriptive text, actions, and non-dialogue content"
            }},
            "character_name": {{
                "gender": "male/female/unknown",
                "lines": <number of dialogue lines>,
                "type": "main/supporting/minor",
                "description": "brief character description"
            }}
        }}
        
        Text to analyze:
        {text_sample}
        
        ALWAYS include "Narrator" as the first entry for narrative/descriptive text.
        """
        
        # Retry logic for Gemini API (handles 503 errors)
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Use the google-genai client to generate content
                response = self.client.models.generate_content(model=self.model_name, contents=prompt)
                # The response object exposes `.text` for the generated text
                response_text = getattr(response, 'text', '').strip()
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"   Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Error analyzing speakers: {e}")
                    # Return default narrator if analysis fails
                    return {
                        "Narrator": {
                            "gender": "unknown",
                            "lines": 1000,
                            "type": "narrator",
                            "description": "Story narrator"
                        }
                    }
        
        try:
            
            # Extract JSON from response (handle markdown formatting)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end]
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end]
            
            speakers = json.loads(response_text)
            
            # Ensure narrator exists and has proper line count
            if "Narrator" not in speakers:
                speakers["Narrator"] = {
                    "gender": "neutral",
                    "lines": 1000,  # Assume narrator has many lines
                    "type": "narrator",
                    "description": "Story narrator"
                }
            else:
                # If Narrator exists but has 0 or missing lines, give it a high count
                narrator_info = speakers["Narrator"]
                lines_count = narrator_info.get('lines', 0)
                try:
                    lines_count = int(lines_count) if lines_count else 0
                except (ValueError, TypeError):
                    lines_count = 0
                
                if lines_count == 0:
                    # Narrator should have many lines - it's all narrative text
                    narrator_info['lines'] = 1000
                    print("⚠️  Narrator had 0 lines in Gemini response, setting to 1000")
            
            # Cache the result
            self._save_to_cache(cache_key, "speakers", speakers)
            
            return speakers
            
        except Exception as e:
            print(f"Error analyzing speakers: {e}")
            # Return default narrator if analysis fails
            return {
                "Narrator": {
                    "gender": "unknown",
                    "lines": 1000,
                    "type": "narrator",
                    "description": "Story narrator"
                }
            }
    
    def chunk_text(self, text: str, speaker_analysis: Dict[str, Any], simple_mode: bool = False) -> List[Dict[str, Any]]:
        """Chunk text intelligently by dialogue and narrative segments
        
        Args:
            text: The full text to chunk
            speaker_analysis: Dictionary of speaker information
            simple_mode: If True, skip Gemini analysis and just split by size (for single speaker mode)
        """
        cache_key = self._get_cache_key(text + json.dumps(speaker_analysis, sort_keys=True))
        
        # Try to load from cache
        cached_chunks = self._load_from_cache(cache_key, "chunks")
        if cached_chunks:
            return cached_chunks
        
        chunks = []
        chunk_id = 0
        
        # For simple mode (single speaker), just split into reasonable chunks without Gemini
        if simple_mode:
            print("📝 Creating text chunks for single speaker mode (no speaker analysis needed)...")
            
            # Split by sentences/paragraphs into ~3000 character chunks
            chunk_size = 3000
            current_pos = 0
            
            while current_pos < len(text):
                chunk_end = min(current_pos + chunk_size, len(text))
                
                # Try to find a good break point (period, newline) if not at end
                if chunk_end < len(text):
                    # Look backwards for a sentence ending
                    search_start = max(current_pos + chunk_size - 500, current_pos)
                    last_period = text.rfind('.', search_start, chunk_end)
                    last_newline = text.rfind('\n', search_start, chunk_end)
                    
                    # Use whichever is later (closer to desired chunk size)
                    break_point = max(last_period, last_newline)
                    
                    if break_point > search_start:
                        chunk_end = break_point + 1
                
                chunk_text_content = text[current_pos:chunk_end].strip()
                
                if chunk_text_content:
                    chunks.append({
                        'id': chunk_id,
                        'text': chunk_text_content,
                        'speaker': 'Narrator',
                        'start_pos': current_pos,
                        'end_pos': chunk_end,
                        'speakers': ['Narrator']
                    })
                    chunk_id += 1
                
                current_pos = chunk_end
            
            print(f"✅ Created {len(chunks)} text chunks for single speaker mode")
        else:
            # Multi-speaker mode: Use Gemini to break entire text into dialogue segments
            print("📝 Analyzing text structure and dialogue...")
            
            # Process in manageable sections (Gemini has token limits)
            section_size = 8000  # Characters per section
            current_pos = 0
            
            while current_pos < len(text):
                section_end = min(current_pos + section_size, len(text))
                section_text = text[current_pos:section_end]
                
                # Analyze this section
                chunk_info = self._analyze_chunk_speaker(section_text, speaker_analysis)
                
                # Debug: show what we got back
                print(f"  📊 Section {current_pos}-{section_end}: chunk_info has {len(chunk_info.get('segments', []))} segments")
                
                # If we got detailed segments from Gemini, use them
                if 'segments' in chunk_info and chunk_info['segments']:
                    for segment in chunk_info['segments']:
                        segment_text = segment.get('text', '').strip()
                        segment_speaker = segment.get('speaker', 'Narrator')
                        
                        if segment_text:  # Only add non-empty segments
                            chunks.append({
                                'id': chunk_id,
                                'text': segment_text,
                                'speaker': segment_speaker,
                                'start_pos': current_pos,
                                'end_pos': current_pos + len(segment_text),
                                'speakers': [segment_speaker]
                            })
                            chunk_id += 1
                    # Move to next section (don't try to track position by segment length)
                    current_pos = section_end
            else:
                # Fallback: treat entire section as one chunk
                chunks.append({
                    'id': chunk_id,
                    'text': section_text.strip(),
                    'speaker': chunk_info['primary_speaker'],
                    'start_pos': current_pos,
                    'end_pos': section_end,
                    'speakers': chunk_info['all_speakers']
                })
                chunk_id += 1
                current_pos = section_end
            
            print(f"  Processed {min(current_pos, len(text))}/{len(text)} characters...")
        
        # Cache the result
        self._save_to_cache(cache_key, "chunks", chunks)
        
        print(f"✅ Created {len(chunks)} dialogue/narrative segments")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved)
        sentence_endings = re.compile(r'[.!?]+[\s\'"]*')
        sentences = sentence_endings.split(text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _analyze_chunk_speaker(self, chunk_text: str, speaker_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a chunk to determine primary speaker using Gemini"""
        # Quick check: if there's no dialogue, it's all narrator
        # Support both standard quotes (") and smart/curly quotes ("")
        # Use actual Unicode characters for smart quotes
        smart_left_double = chr(8220)  # "
        smart_right_double = chr(8221)  # "
        dialogue_pattern = re.compile(f'[{smart_left_double}"]([^{smart_right_double}"]*?)[{smart_right_double}"]')
        dialogue_matches = dialogue_pattern.findall(chunk_text)
        
        if not dialogue_matches:
            return {
                'primary_speaker': "Narrator",
                'all_speakers': ["Narrator"]
            }
        
        # Use Gemini to analyze the chunk and identify speakers
        known_characters = [name for name in speaker_analysis.keys() if name != "Narrator"]
        
        # Use MORE of the chunk text (up to 6000 chars instead of 3000)
        chunk_sample = chunk_text[:6000]
        
        prompt = f"""You are analyzing a narrative text to identify dialogue and speakers. Break this text into small segments, with each dialogue line as its own segment.

Known characters in this book: {', '.join(known_characters)}

CRITICAL INSTRUCTIONS:
1. ONLY segment the text provided below - do NOT add any text that isn't in the input
2. Text in "quotation marks" = dialogue spoken by a character
3. Text NOT in quotation marks = "Narrator" 
4. For dialogue, identify WHO is speaking by:
   - Looking at dialogue tags (e.g., "said Maeve", "Vitor asks", "she replied")
   - Using context from surrounding text (if character just performed an action, they likely speak next)
   - Tracking conversation flow (speakers alternate in dialogue)
5. Keep dialogue text WITH its quotation marks
6. Each dialogue line should be its own segment
7. Narrative paragraphs can be grouped together
8. When dialogue has no explicit tag, infer the speaker from context and conversation flow

EXAMPLE:
Text: I walked into the room. "Hello there," Mary said cheerfully. "How are you?" I smiled. "I'm doing well, thanks for asking," I replied.

Correct Output:
{{
  "segments": [
    {{"text": "I walked into the room.", "speaker": "Narrator"}},
    {{"text": "\\"Hello there,\\" Mary said cheerfully.", "speaker": "Mary"}},
    {{"text": "\\"How are you?\\"", "speaker": "Mary"}},
    {{"text": "I smiled.", "speaker": "Narrator"}},
    {{"text": "\\"I'm doing well, thanks for asking,\\" I replied.", "speaker": "Narrator"}}
  ]
}}

Now analyze this text:
{chunk_sample}

Return ONLY a JSON object in this exact format:
{{
  "segments": [
    {{"text": "...", "speaker": "..."}},
    ...
  ]
}}"""

        # Retry logic for Gemini API (handles 503 errors)
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(model=self.model_name, contents=prompt)
                response_text = getattr(response, 'text', '').strip()
                
                print(f"🔍 Gemini dialogue segmentation response (first 500 chars): {response_text[:500]}")
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"   Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    raise e
        
        try:
            
            # Extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end]
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end]
            
            result = json.loads(response_text.strip())
            segments = result.get('segments', [])

            # Normalize speaker names in the returned segments to match known characters
            for seg in segments:
                raw_sp = seg.get('speaker', 'Narrator')
                norm_sp = self._normalize_speaker_name(raw_sp, known_characters)
                seg['speaker'] = norm_sp

            print(f"✅ Got {len(segments)} dialogue segments from Gemini")
            # Debug: show a quick preview of the first few segments
            try:
                preview = [(s.get('speaker'), s.get('text','')[:80].replace('\n',' ')) for s in segments[:6]]
                print(f"   Segments preview: {preview}")
            except Exception:
                pass
            
            # Count speakers to find primary
            speaker_counts = {}
            all_speakers = set()
            for segment in segments:
                speaker = segment.get('speaker', 'Narrator')
                all_speakers.add(speaker)
                # Count by text length, not just number of segments
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + len(segment.get('text', ''))
            
            # Primary speaker is the one with most text (excluding Narrator)
            non_narrator_speakers = {k: v for k, v in speaker_counts.items() if k != "Narrator"}
            if non_narrator_speakers:
                primary_speaker = max(non_narrator_speakers.items(), key=lambda x: x[1])[0]
            else:
                primary_speaker = "Narrator"
            
            return {
                'primary_speaker': primary_speaker,
                'all_speakers': list(all_speakers),
                'segments': segments
            }
            
        except Exception as e:
            print(f"❌ Error analyzing chunk speaker: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to narrator
            return {
                'primary_speaker': "Narrator",
                'all_speakers': ["Narrator"]
            }
    
    def get_cached_analysis(self, book_path: Path) -> Optional[Dict[str, Any]]:
        """Get cached speaker analysis for a book"""
        text = self.extract_text(str(book_path))
        cache_key = self._get_cache_key(text)
        return self._load_from_cache(cache_key, "speakers")
    
    def get_cached_chunks(self, book_path: Path) -> Optional[List[Dict[str, Any]]]:
        """Get cached chunks for a book"""
        text = self.extract_text(str(book_path))
        speaker_analysis = self.get_cached_analysis(book_path)
        if speaker_analysis:
            cache_key = self._get_cache_key(text + json.dumps(speaker_analysis, sort_keys=True))
            return self._load_from_cache(cache_key, "chunks")
        return None

    def _normalize_speaker_name(self, raw_name: str, known_characters: List[str]) -> str:
        """Normalize a speaker name returned by Gemini to one of the known characters.

        - Strips surrounding parentheses/quotes and whitespace
        - Case-insensitive exact match to known characters
        - Partial/containment match if exact not found
        - Falls back to 'Narrator' for empty/unknown values
        """
        if not raw_name:
            return "Narrator"

        # Remove parentheses and surrounding quotes
        name = raw_name.strip()
        # remove surrounding parentheses like (Narrator)
        if name.startswith("(") and name.endswith(")"):
            name = name[1:-1].strip()
        # remove surrounding quotes
        # If the name is wrapped in matching single or double quotes, unwrap them
        if (name.startswith('"') and name.endswith('"')) or (name.startswith("'") and name.endswith("'")):
            name = name[1:-1].strip()
        name = name.strip()

        if not name:
            return "Narrator"

        # Try exact (case-insensitive) match
        lower_name = name.lower()
        for kc in known_characters:
            if kc.lower() == lower_name:
                return kc

        # Try partial contains
        for kc in known_characters:
            if lower_name in kc.lower() or kc.lower() in lower_name:
                return kc

        # If no match, title-case single-word names (best-effort) or return original trimmed
        if ' ' not in name:
            return name.strip().title()

        return name