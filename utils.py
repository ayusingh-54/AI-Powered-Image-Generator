"""
Utility functions for AI Image Generator
"""

import os
import json
import re
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import config

def sanitize_filename(text, max_length=50):
    """
    Convert text prompt to safe filename
    
    Args:
        text: Input text prompt
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename string
    """
    # Remove special characters
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    # Replace spaces with underscores
    text = text.replace(' ', '_')
    # Limit length
    text = text[:max_length]
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{text}_{timestamp}"

def create_output_directory(base_dir=None):
    """
    Create output directory with timestamp
    
    Args:
        base_dir: Base directory path (default: config.OUTPUT_DIR)
        
    Returns:
        Path to created directory
    """
    if base_dir is None:
        base_dir = config.OUTPUT_DIR
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def save_image_with_metadata(image, prompt, output_path, metadata=None):
    """
    Save image with associated metadata
    
    Args:
        image: PIL Image object
        prompt: Text prompt used
        output_path: Full path including filename
        metadata: Additional metadata dictionary
        
    Returns:
        Saved file path
    """
    # Save image
    image.save(output_path, quality=95)
    
    # Save metadata if enabled
    if config.SAVE_METADATA and metadata:
        metadata_path = output_path.replace('.png', '.json').replace('.jpg', '.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return output_path

def add_watermark(image, text=None, opacity=None):
    """
    Add watermark to image
    
    Args:
        image: PIL Image object
        text: Watermark text (default: config.WATERMARK_TEXT)
        opacity: Opacity value 0-1 (default: config.WATERMARK_OPACITY)
        
    Returns:
        Image with watermark
    """
    if not config.ENABLE_WATERMARK:
        return image
    
    if text is None:
        text = config.WATERMARK_TEXT
    if opacity is None:
        opacity = config.WATERMARK_OPACITY
    
    # Create a copy
    img_with_watermark = image.copy()
    
    # Create watermark layer
    watermark = Image.new('RGBA', img_with_watermark.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    
    # Try to use a nice font, fall back to default
    try:
        font_size = max(20, img_with_watermark.size[0] // 40)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position (bottom right)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = img_with_watermark.size[0] - text_width - 10
    y = img_with_watermark.size[1] - text_height - 10
    
    # Draw semi-transparent text
    text_color = (255, 255, 255, int(255 * opacity))
    draw.text((x, y), text, fill=text_color, font=font)
    
    # Composite watermark onto image
    img_with_watermark = img_with_watermark.convert('RGBA')
    img_with_watermark = Image.alpha_composite(img_with_watermark, watermark)
    
    return img_with_watermark.convert('RGB')

def filter_inappropriate_content(prompt):
    """
    Check prompt for inappropriate content
    
    Args:
        prompt: Text prompt to check
        
    Returns:
        tuple: (is_safe: bool, filtered_prompt: str, warning: str)
    """
    if not config.ENABLE_CONTENT_FILTER:
        return True, prompt, None
    
    prompt_lower = prompt.lower()
    
    # Check for prohibited keywords
    for keyword in config.PROHIBITED_KEYWORDS:
        if keyword in prompt_lower:
            warning = f"⚠️ Content filter triggered: Prompt contains potentially inappropriate content ('{keyword}')"
            return False, "", warning
    
    return True, prompt, None

def format_generation_time(seconds):
    """
    Format generation time in human-readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes} min {remaining_seconds} sec"

def estimate_generation_time(num_images, num_steps):
    """
    Estimate generation time based on hardware and parameters
    
    Args:
        num_images: Number of images to generate
        num_steps: Number of inference steps
        
    Returns:
        Estimated time in seconds
    """
    if config.USE_GPU:
        # GPU estimation: ~0.3 seconds per step
        time_per_image = num_steps * 0.3
    else:
        # CPU estimation: ~10 seconds per step
        time_per_image = num_steps * 10
    
    return num_images * time_per_image

def create_metadata_dict(prompt, negative_prompt, params):
    """
    Create metadata dictionary for generated image
    
    Args:
        prompt: Text prompt
        negative_prompt: Negative prompt
        params: Generation parameters dictionary
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "parameters": params,
        "timestamp": datetime.now().isoformat(),
        "model": config.MODEL_ID,
        "device": config.DEVICE,
        "generator": "AI Image Generator v1.0",
        "generated_by": "Stable Diffusion"
    }
    return metadata

def log_generation(prompt, success, error_msg=None):
    """
    Log generation attempt
    
    Args:
        prompt: Text prompt
        success: Whether generation succeeded
        error_msg: Error message if failed
    """
    if not config.ENABLE_LOGGING:
        return
    
    timestamp = datetime.now().isoformat()
    status = "SUCCESS" if success else "FAILED"
    log_entry = f"[{timestamp}] {status} - Prompt: {prompt[:100]}"
    
    if error_msg:
        log_entry += f" | Error: {error_msg}"
    
    log_entry += "\n"
    
    try:
        with open(config.LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")

def get_prompt_suggestions():
    """
    Return list of example prompts for users
    
    Returns:
        List of example prompt strings
    """
    return [
        "a futuristic city at sunset, highly detailed, 4K",
        "portrait of a robot in Van Gogh style, oil painting",
        "a serene mountain landscape with a lake, professional photography",
        "a mystical forest with glowing mushrooms, fantasy art",
        "a cute cat wearing a space suit, digital art",
        "an ancient castle on a cliff during a storm, dramatic lighting",
        "a steampunk airship flying through clouds, detailed mechanical parts",
        "a cyberpunk street scene with neon lights, rainy night",
        "a magical library with floating books, warm lighting",
        "a dragon perched on a mountain peak, epic fantasy illustration"
    ]

def convert_image_format(image, format_name):
    """
    Convert PIL Image to specified format
    
    Args:
        image: PIL Image object
        format_name: Target format (PNG, JPEG, WEBP)
        
    Returns:
        Converted image
    """
    if format_name.upper() == "JPEG" and image.mode == "RGBA":
        # JPEG doesn't support transparency
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3] if len(image.split()) == 4 else None)
        return rgb_image
    return image

def create_image_grid(images, grid_size=None):
    """
    Create a grid of images
    
    Args:
        images: List of PIL Image objects
        grid_size: Tuple (rows, cols) or None for auto
        
    Returns:
        Combined grid image
    """
    if not images:
        return None
    
    num_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        cols = min(2, num_images)
        rows = (num_images + cols - 1) // cols
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    
    # Get dimensions from first image
    img_width, img_height = images[0].size
    
    # Create grid
    grid_width = cols * img_width
    grid_height = rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Place images
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        grid_image.paste(img, (x, y))
    
    return grid_image
