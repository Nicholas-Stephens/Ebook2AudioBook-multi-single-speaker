#!/bin/bash
# Entrypoint script for Ebook2Audio Docker container
# Checks for required models and downloads if missing, then starts the app

set -e

CHECKPOINT_DIR="/app/index-tts/checkpoints"
REQUIRED_FILE="gpt.pth"  # Main model file to check (3.3GB)

echo "=============================================="
echo "🎧 Ebook2Audio - Multi/Single Speaker"
echo "=============================================="

# Check if models exist
if [ ! -f "$CHECKPOINT_DIR/$REQUIRED_FILE" ]; then
    echo ""
    echo "📦 IndexTTS2 models not found. Downloading..."
    echo "   This may take 10-30 minutes depending on your connection."
    echo "   Models will be saved to: $CHECKPOINT_DIR"
    echo ""
    
    # Run the download script
    python3 /app/download_models.py "$CHECKPOINT_DIR"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Models downloaded successfully!"
    else
        echo ""
        echo "⚠️  Model download may have failed. The app will try to continue..."
        echo "   If TTS doesn't work, check your internet connection and restart."
    fi
else
    echo "✅ IndexTTS2 models found at $CHECKPOINT_DIR"
fi

# Check for config.yaml (required for model initialization)
if [ ! -f "$CHECKPOINT_DIR/config.yaml" ]; then
    echo "⚠️  config.yaml not found - downloading..."
    python3 /app/download_models.py "$CHECKPOINT_DIR"
fi

echo ""
echo "🚀 Starting Ebook2Audio application..."
echo ""

# Start the main application
exec python3 /app/app.py
