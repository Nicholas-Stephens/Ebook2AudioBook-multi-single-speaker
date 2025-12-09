# 🚀 Quick Start Guide

Get your first multi-voice audiobook running in under 30 minutes!

**🆕 NEW**: Fast Preview Mode for CPU development - 2-3x faster testing!

## ⏱️ Time Required

- Setup: 15-20 minutes (first time only)
- Voice Samples: 5-10 minutes
- Book Processing: 
  - **CPU + Fast Preview**: 10-20 minutes (testing quality)
  - **CPU (default)**: 30-60 minutes (high quality, slow)
  - **GPU**: 1-5 minutes (high quality, fast)

## 📋 Checklist

- [ ] Python 3.10+ installed
- [ ] 16GB+ RAM available
- [ ] 20GB+ free disk space
- [ ] Internet connection for downloads
- [ ] Gemini API key ([Get free key](https://makersuite.google.com/app/apikey))
- [ ] Audio samples for voices (or use temporary samples)

## 🎬 Step-by-Step

### 1. Clone and Setup (5 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/Ebook2MultiSpeaker.git
cd Ebook2MultiSpeaker

# Clone IndexTTS
git clone https://github.com/index-tts/index-tts.git
cd index-tts
git lfs pull
cd ..

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models (15 minutes, one-time)

```bash
# This downloads ~5-10GB of AI models
python3 setup_indextts.py

# If in China, use ModelScope mirror:
# python3 setup_indextts.py --source modelscope
```

☕ **Grab a coffee!** This takes 15-30 minutes on first run.

### 3. Prepare Voice Samples (5 minutes)

**Option A: Use Your Own Voices** (Recommended)
```bash
# Record or find 10-15 second voice samples
# Save as: voice_samples/CharacterName.wav

# Example:
voice_samples/
├── Narrator.wav    # Record yourself reading
├── Hero.wav        # Friend's voice or AI-generated
└── Villain.wav     # Different person or AI voice
```

**Option B: Quick Test with Placeholder**
```bash
# For testing, you can use any short audio file
cp any_audio.mp3 voice_samples/TestVoice.wav
```

**Voice Sample Tips:**
- 📏 10-15 seconds ideal
- 🎤 Clear speech, no background noise
- 💬 Natural conversation tone
- ✅ WAV or MP3 format

### 4. Add Your Book (1 minute)

```bash
# Copy your ebook to the books folder
cp path/to/your/book.epub books/

# Supported: .txt, .pdf, .epub
```

### 5. Set API Key (1 minute)

```bash
# Get free key: https://makersuite.google.com/app/apikey

# Linux/Mac:
export GEMINI_API_KEY='your-api-key-here'

# Windows PowerShell:
$env:GEMINI_API_KEY='your-api-key-here'

# Or add to config.py:
GEMINI_API_KEY = 'your-api-key-here'
```

### 6. Launch! (30 seconds)

```bash
python3 app.py
```

Open your browser: **http://127.0.0.1:7860**

## 🎯 Using the Web Interface

### Step 1: Load Voice Samples
1. Click "🔄 Reload Voices" to see your voice samples
2. Verify they're listed in the "Available Voice Samples" field

### Step 2: Select Book
1. Choose your book from the dropdown
2. If not visible, click "🔄 Refresh"

### Step 3: Configure Settings (Optional)
- **Minor character threshold:** 5 (recommended)
- **Assign minor to narrator:** ✓ (recommended)

### Step 4: Analyze Book
1. Click "🔍 Analyze Book"
2. Wait 1-2 minutes for AI analysis
3. Review detected characters

### Step 5: Assign Voices
- Auto-assigned voices shown for each character
- Click dropdowns to change any assignment
- Or click "🤖 Auto-Assign Voices" to regenerate

### Step 6: Generate Audiobook
1. Click "🎵 Generate Audiobook"
2. Monitor progress in terminal
3. Find completed audiobook in `output/final/`

**Generation Time:**
- GPU: 5-15 min per hour of audio
- CPU: 30-60 min per hour of audio

## 📂 Where to Find Output

```
output/
├── chunks/              # Temporary audio pieces
└── final/
    └── YourBook_timestamp.wav  # ← YOUR AUDIOBOOK! 🎉
```

## 🎧 Listen and Enjoy!

Open the WAV file in any audio player:
- VLC Media Player
- Windows Media Player
- macOS QuickTime
- Audacity (for editing)

## 🐛 Quick Troubleshooting

### "No voice samples found"
```bash
# Make sure files are in the right place
ls voice_samples/
# Should show: YourVoice.wav

# Click "🔄 Reload Voices" in the UI
```

### "IndexTTS not available"
```bash
# Re-run setup
python3 setup_indextts.py

# Check models downloaded
ls index-tts/checkpoints/
```

### "CUDA out of memory"
- Close other GPU applications
- Restart the app
- If persists, use CPU mode (slower but works)

### "Gemini API error"
```bash
# Verify your API key is set
echo $GEMINI_API_KEY

# Or check it's working:
python3 test_api_key.py
```

### Slow Generation
- **Normal for CPU mode:** 30-60 min/hour
- **GPU available?** Check: `nvidia-smi`
- **RAM sufficient?** Check: `free -h` (Linux) or Task Manager (Windows)

## 🚀 Docker Quick Start

If you prefer Docker:

```bash
# Create environment file
echo "GEMINI_API_KEY=your-key" > .env

# GPU mode (recommended)
docker-compose up -d

# OR CPU mode
docker-compose --profile cpu up -d

# View logs
docker-compose logs -f

# Open browser: http://localhost:7860
```

See [DOCKER_DEPLOY.md](DOCKER_DEPLOY.md) for detailed Docker guide.

## 💡 Pro Tips

1. **Good Voice Samples = Good Results**
   - Invest time in quality 10-15 second samples
   - Clear, natural speech
   - Minimal background noise

2. **Start Small**
   - Test with a short story or chapter first
   - Verify quality before processing full book

3. **GPU is Worth It**
   - 10-50x faster than CPU
   - Consider cloud GPU if needed

4. **Reuse Voices**
   - Once you have good samples, save them
   - Use same voices across multiple books

5. **Character Threshold**
   - Set to 5-10 lines for most books
   - Prevents voice-wasting on minor characters

## 📚 Example Workflow

```bash
# 1. One-time setup (first time only)
python3 setup_indextts.py

# 2. Prepare voices (once, reuse for all books)
# Record 3-5 voice samples, save to voice_samples/

# 3. For each book:
cp my_book.epub books/
python3 app.py
# → Analyze → Assign → Generate → Enjoy!

# 4. Find audiobook in output/final/
ls -lh output/final/*.wav
```

## ⏭️ Next Steps

After your first audiobook:
- Try different voice samples
- Experiment with emotion settings
- Process longer books
- Share with friends! 😄

## 📖 More Information

- [README.md](README.md) - Full documentation
- [DOCKER_DEPLOY.md](DOCKER_DEPLOY.md) - Docker deployment
- [voice_samples/README.md](voice_samples/README.md) - Voice sample guide
- [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) - Technical details

## 🆘 Still Need Help?

1. Check terminal output for errors
2. Read the error message carefully
3. Check README troubleshooting section
4. Open a GitHub issue with:
   - Error message
   - What you were trying to do
   - Your system specs

---

**Ready to create your first multi-voice audiobook?** 🎤📚🎧

**Let's go!** Run: `python3 setup_indextts.py`
