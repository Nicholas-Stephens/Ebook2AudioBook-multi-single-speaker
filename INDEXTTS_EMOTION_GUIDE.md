# IndexTTS Emotion Control Guide

This document explains how IndexTTS-2's emotion control system works and how to use it in the audiobook generator.

## Overview

IndexTTS-2 uses a sophisticated emotion system that can:
- **Auto-detect emotions** from text using Qwen AI
- **Manually control emotions** via emotion vectors
- **Blend emotions** from reference audio with emotion alpha parameter
- Support 8 distinct emotion categories

## Emotion Categories

IndexTTS-2 supports 8 emotion types:

| English | Chinese | Description |
|---------|---------|-------------|
| `happy` | 高兴 | Joyful, cheerful, excited |
| `angry` | 愤怒 | Mad, furious, irritated |
| `sad` | 悲伤 | Sorrowful, mournful, crying |
| `afraid` | 恐惧 | Scared, fearful, terrified |
| `disgusted` | 反感 | Repulsed, revolted, disgusted |
| `melancholic` | 低落 | Down, depressed, gloomy |
| `surprised` | 惊讶 | Shocked, amazed, startled |
| `calm` | 自然 | Neutral, natural, relaxed |

## How Emotion Detection Works

### 1. Automatic Text-Based Detection

When you set `use_emo_text=True`, IndexTTS uses the **Qwen AI model** to analyze the text and automatically detect emotions:

```python
tts.infer(
    spk_audio_prompt="voice.wav",
    text="I'm so excited to see you!",
    use_emo_text=True,  # Enable automatic emotion detection
    output_path="output.wav"
)
```

**How it works:**
1. Text is sent to Qwen emotion classifier
2. Qwen returns emotion scores (0.0 to 1.2 for each emotion)
3. Scores are automatically converted to emotion vectors
4. TTS synthesizes audio with those emotions

**Example detection:**
```
Input: "I can't believe this happened!"
Output: {
    "happy": 0.0,
    "angry": 0.3,
    "sad": 0.0,
    "afraid": 0.0,
    "disgusted": 0.0,
    "melancholic": 0.0,
    "surprised": 0.9,
    "calm": 0.1
}
```

### 2. Manual Emotion Vectors

You can manually specify emotion intensities using **emotion vectors** - a list of 8 float values in this exact order:

```python
# Order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
emotion_vector = [0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0]  # Sad + slightly melancholic

tts.infer(
    spk_audio_prompt="voice.wav",
    text="I miss you so much...",
    emo_vector=emotion_vector,  # Use manual emotion control
    output_path="output.wav"
)
```

**Valid range:** 0.0 to 1.2 for each emotion (clamped automatically)

**Common presets:**
```python
# Excited/Happy
happy_emotion = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Angry/Frustrated
angry_emotion = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Sad/Crying
sad_emotion = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Scared/Terrified
afraid_emotion = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

# Calm/Neutral (default if no emotions)
calm_emotion = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

# Mixed emotions (e.g., bittersweet)
bittersweet = [0.3, 0.0, 0.4, 0.0, 0.0, 0.3, 0.0, 0.0]

# Dramatic surprise
dramatic = [0.0, 0.2, 0.0, 0.3, 0.0, 0.0, 0.8, 0.0]
```

### 3. Reference Audio Emotion

You can use a separate **emotion reference audio** to capture emotion from a voice recording:

```python
tts.infer(
    spk_audio_prompt="narrator_voice.wav",  # Base voice
    emo_audio_prompt="crying_sample.wav",   # Emotion reference
    emo_alpha=0.7,  # 70% emotion reference, 30% base
    text="This is heartbreaking...",
    output_path="output.wav"
)
```

**How it works:**
- `spk_audio_prompt`: Defines the voice timbre/tone
- `emo_audio_prompt`: Defines the emotional expression
- `emo_alpha`: Blending factor (0.0 to 1.0)
  - `0.0` = No emotion from reference (use speaker's natural emotion)
  - `0.5` = 50/50 blend
  - `1.0` = Full emotion from reference

## Parameters Reference

### Core Emotion Parameters

```python
tts.infer(
    spk_audio_prompt: str,              # Required: Path to speaker voice sample
    text: str,                          # Required: Text to synthesize
    output_path: str,                   # Required: Output file path
    
    # Emotion control (pick ONE method):
    emo_audio_prompt: str = None,       # Path to emotion reference audio
    emo_alpha: float = 1.0,             # Emotion blend factor (0.0-1.0)
    emo_vector: List[float] = None,     # Manual emotion vector [8 values]
    use_emo_text: bool = False,         # Auto-detect emotions from text
    emo_text: str = None,               # Custom text for emotion detection
    
    # Other parameters:
    use_random: bool = False,           # Randomize emotion slightly
    interval_silence: int = 200,        # Silence between segments (ms)
    verbose: bool = False,              # Print debug info
    stream_return: bool = False,        # Stream audio chunks
    **generation_kwargs                 # Additional generation parameters
)
```

### Emotion Priority

When multiple emotion methods are specified, IndexTTS uses this priority:

1. **Emotion Vector** (`emo_vector`) - Highest priority, manual control
2. **Text-based detection** (`use_emo_text=True`) - Auto-detect from text
3. **Reference audio** (`emo_audio_prompt`) - Emotion from audio sample
4. **Default**: Uses speaker's natural emotion from `spk_audio_prompt`

**Important:** If you use `emo_vector` or `use_emo_text=True`, the `emo_audio_prompt` is automatically ignored!

## Integration with Audiobook Generator

### Current Implementation

In `voice_processor_indextts.py`, the current implementation uses:

```python
def generate_chunk_audio(self, text, voice_name, chunk_id, book_name, generation_kwargs=None):
    result = self.tts_model.infer(
        spk_audio_prompt=voice_sample_path,
        text=text,
        output_path=str(output_file),
        emo_audio_prompt=voice_sample_path,  # Uses same sample for emotion
        emo_vector=None,                     # No manual control
        emo_alpha=1.0,                       # Full emotion from audio
        verbose=False,
        **generation_kwargs,
    )
```

This uses the voice sample for both timbre AND emotion (simple approach).

### Adding Emotion Control

We can add emotion control in several ways:

#### Option 1: Per-Chunk Emotion Detection (Automatic)

Analyze each chunk's text and apply appropriate emotions:

```python
def generate_chunk_audio(self, text, voice_name, chunk_id, book_name, 
                        auto_emotion=True, generation_kwargs=None):
    gen_kwargs = generation_kwargs.copy() if generation_kwargs else {}
    
    if auto_emotion:
        # Let IndexTTS auto-detect emotions from the text
        gen_kwargs['use_emo_text'] = True
    
    result = self.tts_model.infer(
        spk_audio_prompt=voice_sample_path,
        text=text,
        output_path=str(output_file),
        emo_audio_prompt=None if auto_emotion else voice_sample_path,
        emo_alpha=1.0,
        verbose=False,
        **gen_kwargs,
    )
```

#### Option 2: Manual Scene-Based Emotions

Allow users to specify emotions for different scenes:

```python
emotion_map = {
    "action": [0.2, 0.4, 0.0, 0.3, 0.0, 0.0, 0.5, 0.0],  # Excited + surprised + angry
    "sad": [0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0],     # Sad + melancholic
    "happy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],   # Happy + calm
    "tense": [0.0, 0.3, 0.0, 0.6, 0.0, 0.0, 0.2, 0.0],   # Afraid + angry + surprised
    "neutral": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Calm
}

def generate_chunk_audio(self, text, voice_name, chunk_id, book_name, 
                        emotion_type="neutral", generation_kwargs=None):
    gen_kwargs = generation_kwargs.copy() if generation_kwargs else {}
    
    if emotion_type in emotion_map:
        gen_kwargs['emo_vector'] = emotion_map[emotion_type]
    
    result = self.tts_model.infer(
        spk_audio_prompt=voice_sample_path,
        text=text,
        output_path=str(output_file),
        **gen_kwargs,
    )
```

#### Option 3: Dialogue-Aware Emotions

Automatically adjust emotions based on dialogue tags:

```python
def detect_emotion_from_dialogue(text):
    """Detect emotion from dialogue tags and punctuation"""
    text_lower = text.lower()
    
    # Check for exclamations
    if text.count('!') > 0:
        if any(word in text_lower for word in ['no', 'stop', 'never']):
            return [0.0, 0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0]  # Angry + afraid
        else:
            return [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0]  # Happy + surprised
    
    # Check for questions with concern
    if '?' in text and any(word in text_lower for word in ['what', 'why', 'how']):
        return [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.5, 0.2]  # Surprised + afraid + calm
    
    # Check for sadness markers
    if any(word in text_lower for word in ['cry', 'sob', 'tear', 'miss', 'lost']):
        return [0.0, 0.0, 0.7, 0.0, 0.0, 0.3, 0.0, 0.0]  # Sad + melancholic
    
    # Default: Let IndexTTS auto-detect
    return None  # None means use auto-detection
```

## Best Practices

### 1. Start with Auto-Detection

For most audiobooks, automatic emotion detection works well:

```python
generation_kwargs = {'use_emo_text': True}
generate_chunk_audio(text, voice, chunk_id, book_name, generation_kwargs=generation_kwargs)
```

### 2. Use Reference Audio for Specific Moods

Collect emotion reference samples for different moods:
- `happy_ref.wav` - Laughing, cheerful voice
- `sad_ref.wav` - Crying, sobbing voice
- `angry_ref.wav` - Shouting, frustrated voice

### 3. Blend Emotions Carefully

Don't max out multiple emotions simultaneously - keep total < 2.0:
```python
# Good: Clear primary emotion with hints of others
[0.7, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0]

# Avoid: Too many strong emotions at once
[1.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0]  # Overwhelming
```

### 4. Test with Different Alpha Values

Experiment with `emo_alpha` for subtle variations:
- `0.3` - Subtle emotional hint
- `0.5` - Moderate emotion
- `0.7` - Strong emotion
- `1.0` - Full emotional expression

### 5. Scene-Based Emotion Mapping

For longer books, map scenes to emotions:
```python
scene_emotions = {
    "chapter_1": "calm",      # Introduction
    "chapter_2": "tense",     # Rising action
    "chapter_3": "action",    # Climax
    "chapter_4": "sad",       # Aftermath
    "chapter_5": "happy"      # Resolution
}
```

## Performance Considerations

### Speed Impact

Emotion detection adds minimal overhead:
- **Auto-detection (`use_emo_text`)**: +0.5-1s per chunk (Qwen model)
- **Manual vectors (`emo_vector`)**: No overhead (direct)
- **Reference audio (`emo_audio_prompt`)**: +0.1-0.2s per chunk (audio processing)

### Memory Impact

- Qwen emotion model: ~2-4GB VRAM
- Emotion vectors: Negligible
- Reference audio caching: ~50MB per sample

### Recommendations

1. **For CPU users**: Use `emo_vector` (manual) to avoid Qwen overhead
2. **For GPU users**: Use `use_emo_text=True` for best results
3. **For production**: Pre-analyze text and cache emotion vectors

## Example: Complete Implementation

```python
class EmotionController:
    """Helper class for managing emotions in audiobook generation"""
    
    def __init__(self):
        self.emotion_presets = {
            "happy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "angry": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "sad": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "afraid": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "disgusted": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "melancholic": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "surprised": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "calm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    
    def detect_from_text(self, text):
        """Simple rule-based emotion detection"""
        text_lower = text.lower()
        
        # Check for strong emotions
        if any(word in text_lower for word in ['scream', 'yell', 'shout', 'furious']):
            return self.emotion_presets['angry']
        
        if any(word in text_lower for word in ['cry', 'sob', 'weep', 'tears']):
            return self.emotion_presets['sad']
        
        if any(word in text_lower for word in ['laugh', 'smile', 'joy', 'delight']):
            return self.emotion_presets['happy']
        
        # Default to auto-detection
        return None
    
    def blend_emotions(self, primary, secondary, ratio=0.7):
        """Blend two emotion vectors"""
        return [
            primary[i] * ratio + secondary[i] * (1 - ratio)
            for i in range(8)
        ]
    
    def get_generation_kwargs(self, text, mode='auto', custom_vector=None):
        """Get generation kwargs with appropriate emotion settings"""
        kwargs = {}
        
        if mode == 'auto':
            kwargs['use_emo_text'] = True
        elif mode == 'manual' and custom_vector:
            kwargs['emo_vector'] = custom_vector
        elif mode == 'detect':
            detected = self.detect_from_text(text)
            if detected:
                kwargs['emo_vector'] = detected
            else:
                kwargs['use_emo_text'] = True
        
        return kwargs

# Usage in audiobook generation:
emotion_ctrl = EmotionController()

for chunk in chunks:
    gen_kwargs = emotion_ctrl.get_generation_kwargs(
        chunk.text, 
        mode='auto'  # or 'manual' or 'detect'
    )
    
    audio_path = voice_processor.generate_chunk_audio(
        text=chunk.text,
        voice_name=chunk.speaker,
        chunk_id=chunk.id,
        book_name="my_book",
        generation_kwargs=gen_kwargs
    )
```

## Troubleshooting

### Issue: Emotions sound too extreme

**Solution:** Reduce emotion vector values to 0.3-0.7 range instead of 1.0

### Issue: No emotion detected automatically

**Solution:** Check if Qwen model is loaded properly. Fallback to manual vectors.

### Issue: Emotions inconsistent between chunks

**Solution:** Use consistent emotion references or disable `use_random=True`

### Issue: Slow generation with auto-detection

**Solution:** Pre-analyze all text and cache emotion vectors, then use manual mode

## Further Reading

- IndexTTS GitHub: https://github.com/yuanpeng16/IndexTTS
- Emotion-aware TTS research papers
- Voice acting emotional ranges reference

---

**Version:** 1.0  
**Last Updated:** October 2025  
**Compatible with:** IndexTTS-2.0+
