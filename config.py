"""
Configuration settings for AI Image Generator
"""

import os
import torch

# Model Configuration
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # Open-source Stable Diffusion model
MODEL_CACHE_DIR = "./model_cache"

# Device Configuration
def get_device():
    """Automatically detect and return the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

DEVICE = get_device()
USE_GPU = DEVICE == "cuda"

# Image Generation Settings
DEFAULT_IMAGE_SIZE = 512  # 512x512 pixels (standard for SD 1.5)
DEFAULT_NUM_INFERENCE_STEPS = 30  # Balance between quality and speed
DEFAULT_GUIDANCE_SCALE = 7.5  # How closely to follow the prompt
DEFAULT_NUM_IMAGES = 1

# Image size options
IMAGE_SIZE_OPTIONS = {
    "Small (512x512)": 512,
    "Medium (640x640)": 640,
    "Large (768x768)": 768,
}

# Style Presets (prompt modifiers)
STYLE_PRESETS = {
    "None": "",
    "Photorealistic": ", professional photography, highly detailed, 4K, sharp focus, photorealistic",
    "Artistic": ", artistic, beautiful, detailed artwork, high quality art",
    "Digital Art": ", digital art, trending on artstation, highly detailed digital painting",
    "Oil Painting": ", oil painting, fine art, canvas texture, painterly style",
    "Watercolor": ", watercolor painting, soft colors, artistic watercolor style",
    "Cartoon": ", cartoon style, animated, colorful illustration, stylized",
    "Anime": ", anime style, manga art, japanese animation style",
    "Sketch": ", pencil sketch, hand drawn, artistic sketch, line art",
    "3D Render": ", 3D render, CGI, octane render, unreal engine, highly detailed 3D",
    "Vintage": ", vintage style, retro, old photograph, aged paper texture",
    "Cyberpunk": ", cyberpunk style, neon lights, futuristic, sci-fi aesthetic",
    "Fantasy": ", fantasy art, magical, mystical, detailed fantasy illustration",
}

# Default negative prompts for better quality
DEFAULT_NEGATIVE_PROMPT = "blurry, bad quality, low resolution, distorted, deformed, ugly, bad anatomy, watermark, signature, text"

# Storage Settings
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metadata Settings
SAVE_METADATA = True
METADATA_FORMAT = "json"

# Export Settings
SUPPORTED_FORMATS = ["PNG", "JPEG", "WEBP"]
DEFAULT_FORMAT = "PNG"

# Content Filtering
ENABLE_CONTENT_FILTER = True

# Prohibited keywords for content filtering
PROHIBITED_KEYWORDS = [
    "nude", "naked", "nsfw", "explicit", "sexual",
    "violence", "gore", "bloody", "weapon",
    "hate", "racist", "discriminatory",
    # Add more as needed
]

# Watermark Settings
ENABLE_WATERMARK = True
WATERMARK_TEXT = "AI Generated"
WATERMARK_OPACITY = 0.3

# Performance Settings
if USE_GPU:
    # GPU optimizations
    ENABLE_ATTENTION_SLICING = True
    ENABLE_VAE_SLICING = True
    ENABLE_CPU_OFFLOAD = False
else:
    # CPU optimizations
    ENABLE_ATTENTION_SLICING = True
    ENABLE_VAE_SLICING = True
    ENABLE_CPU_OFFLOAD = False
    # CPU mode warning
    print("⚠️  Running in CPU mode - generation will be slower (5-10 minutes per image)")
    print("💡 For faster generation, consider using a system with NVIDIA GPU")

# Logging
ENABLE_LOGGING = True
LOG_FILE = "generation_log.txt"

# UI Settings
UI_TITLE = "🎨 AI Image Generator"
UI_DESCRIPTION = """
Generate stunning images from text descriptions using Stable Diffusion AI.

**How to use:**
1. Enter a detailed text description of the image you want
2. Adjust settings for quality and style
3. Click "Generate Images"
4. Download your creations!

**Tips for better results:**
- Be specific and descriptive
- Add quality keywords like "highly detailed", "4K"
- Use style modifiers for consistent aesthetics
- Experiment with negative prompts to avoid unwanted elements
"""

# Hardware information display
def get_hardware_info():
    """Return hardware information string"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"🚀 GPU Detected: {gpu_name} ({gpu_memory:.1f} GB VRAM)"
    else:
        return "💻 Running on CPU (slower but functional)"

HARDWARE_INFO = get_hardware_info()
