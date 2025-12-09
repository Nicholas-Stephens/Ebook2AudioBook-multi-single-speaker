# Ebook2Audio - Multi-Speaker & Single-Speaker Audiobook Generator

Convert ebooks into professional-quality audiobooks with multi-speaker narration and character voices using AI.

## Features

- **Multi-Speaker Mode**: Analyzes book structure to identify speakers/characters and assigns unique voices
- **Single-Speaker Mode**: Generate audiobooks with a single consistent narrator
- **Character Voice Mapping**: Custom voice samples for specific characters
- **Emotion Control**: Fine-tune emotional expression in the generated speech
- **Docker Support**: Easy deployment with Docker and Docker Compose (GPU & CPU modes)
- **Web Interface**: User-friendly Gradio UI for audiobook generation

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU (optional, CPU mode available)
- Google Gemini API key
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nicholas-Stephens/ebook2audio-single.git
   cd ebook2audio-single
   ```

2. **Download models**
   ```bash
   python3 download_models.py
   ```
   This downloads the IndexTTS2 model (~5.5GB) and other required files.

3. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your Google Gemini API key
   export GEMINI_API_KEY="your-api-key-here"
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the app**
   ```bash
   python3 app.py
   ```
   The web interface will be available at `http://localhost:7860`

## Docker Usage

### GPU Mode (recommended)
```bash
export GEMINI_API_KEY="your-api-key-here"
docker-compose up
```

### CPU Mode
```bash
export GEMINI_API_KEY="your-api-key-here"
docker-compose --profile cpu up
```

## Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `GEMINI_MODEL_NAME`: Gemini model to use (default: `gemini-2.5-flash`)
- `DEVICE`: Device for TTS model (`cuda:0` for GPU, `cpu` for CPU)
- `BOOKS_DIR`: Directory for input books (default: `books/`)

### Model Configuration
- Edit `config.py` to adjust voice settings, chunk sizes, and TTS parameters

## Usage

### Multi-Speaker Mode
1. Upload an ebook (.txt, .epub, .pdf)
2. Click "Analyze Book Structure"
3. The app will:
   - Identify speakers/characters
   - Assign voices based on gender and character type
   - Generate audio for each speaker
4. Download the final multi-speaker audiobook

### Single-Speaker Mode
1. Upload an ebook
2. Select or customize the narrator voice
3. Click "Generate Single Speaker Audiobook"
4. Download the completed audiobook

### Custom Character Voices
1. Record voice samples for specific characters
2. Place `.wav` files in `voice_samples/` directory
3. Name them with the character name (e.g., `Harry_Potter.wav`)
4. The app will automatically match and use custom voices

## Project Structure

```
.
├── app.py                          # Gradio web interface
├── book_analyzer.py                # Book analysis with Gemini API
├── voice_processor_indextts.py     # IndexTTS2 voice processing
├── audio_compiler.py               # Audio merging and compilation
├── download_models.py              # Model download script
├── docker-compose.yml              # Docker Compose configuration
├── Dockerfile                      # Multi-stage Docker build
├── requirements.txt                # Python dependencies
├── index-tts/                      # IndexTTS2 TTS engine
│   ├── indextts/                   # Core TTS implementation
│   └── checkpoints/                # Model checkpoints (5.5GB+)
├── books/                          # Input ebooks directory
├── output/                         # Generated audiobooks
│   ├── multi_final/               # Multi-speaker outputs
│   └── single_final/              # Single-speaker outputs
├── voice_samples/                  # Custom character voices
└── cache/                          # Model and HuggingFace cache
```

## Technical Details

### Models
- **TTS Engine**: IndexTTS2 (Chinese-optimized, supports multi-language)
- **Book Analysis**: Google Gemini API
- **Voice Samples**: Custom voice processing with IndexTTS2

### Audio Processing
- Sample rate: 24kHz
- Bit depth: 16-bit
- Channels: Mono
- Codec: WAV for processing, MP3 for final output

## Troubleshooting

### Model Download Issues
If the app hangs during initialization:
1. Ensure `pinyin.vocab` is present in `index-tts/checkpoints/`
2. Run `python3 download_models.py` manually
3. Check available disk space (requires ~5.5GB minimum)

### GPU Memory Issues
- Reduce chunk size in `config.py`
- Use CPU mode for limited VRAM (under 8GB)
- Process shorter books first to test

### Gemini API Errors
- Verify `GEMINI_API_KEY` is correctly set
- Check API quota and rate limits
- Switch to `gemini-2.5-flash` model (default) for better availability

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **IndexTTS2**: Advanced multi-lingual text-to-speech engine
- **Google Gemini**: Book analysis and speaker identification
- **Gradio**: Web interface framework
- **HuggingFace**: Model hosting and distribution

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Note**: This project requires significant computational resources. GPU acceleration is recommended for faster processing. CPU mode is available but will be significantly slower.
