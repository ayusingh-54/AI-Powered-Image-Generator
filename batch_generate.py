"""
Batch generation script for creating multiple images from a list of prompts
Useful for generating multiple images without the UI
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import time

from image_generator import ImageGenerator, PromptEnhancer
import utils
import config

def load_prompts_from_file(filepath):
    """Load prompts from text or JSON file"""
    path = Path(filepath)
    
    if path.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

def batch_generate(
    prompts,
    output_dir=None,
    num_images_per_prompt=1,
    num_inference_steps=30,
    guidance_scale=7.5,
    style="None",
    use_negative_prompt=True,
    custom_negative_prompt=None
):
    """
    Generate images for a batch of prompts
    
    Args:
        prompts: List of text prompts
        output_dir: Output directory (default: timestamped folder)
        num_images_per_prompt: Number of variations per prompt
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        style: Style preset
        use_negative_prompt: Whether to use negative prompt
        custom_negative_prompt: Custom negative prompt (overrides default)
    """
    # Setup
    if output_dir is None:
        output_dir = utils.create_output_directory()
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Batch Image Generation")
    print("="*70)
    print(f"Total prompts: {len(prompts)}")
    print(f"Images per prompt: {num_images_per_prompt}")
    print(f"Output directory: {output_dir}")
    print(f"Hardware: {config.DEVICE.upper()}")
    print("="*70)
    
    # Initialize generator
    generator = ImageGenerator()
    enhancer = PromptEnhancer()
    
    print("\nLoading model...")
    generator.load_model()
    print("✅ Model loaded\n")
    
    # Statistics
    total_images = len(prompts) * num_images_per_prompt
    successful = 0
    failed = 0
    start_time = time.time()
    
    # Process each prompt
    for prompt_idx, prompt in enumerate(prompts, 1):
        print(f"\n[{prompt_idx}/{len(prompts)}] Processing: '{prompt[:60]}...'")
        
        try:
            # Content filtering
            is_safe, _, warning = utils.filter_inappropriate_content(prompt)
            if not is_safe:
                print(f"  ⚠️  Skipped: {warning}")
                failed += num_images_per_prompt
                continue
            
            # Enhance prompt
            enhanced_prompt = enhancer.enhance_prompt(prompt, style)
            
            # Prepare negative prompt
            negative_prompt = None
            if use_negative_prompt:
                negative_prompt = custom_negative_prompt or config.DEFAULT_NEGATIVE_PROMPT
            
            # Generate
            print(f"  🎨 Generating {num_images_per_prompt} image(s)...")
            images, metadata = generator.generate(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_images=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            # Save images
            prompt_dir = Path(output_dir) / f"prompt_{prompt_idx:03d}"
            prompt_dir.mkdir(exist_ok=True)
            
            for img_idx, img in enumerate(images, 1):
                img_with_watermark = utils.add_watermark(img)
                filename = f"image_{img_idx}.png"
                filepath = prompt_dir / filename
                
                utils.save_image_with_metadata(
                    img_with_watermark,
                    enhanced_prompt,
                    str(filepath),
                    utils.create_metadata_dict(enhanced_prompt, negative_prompt, metadata)
                )
                
                print(f"  ✅ Saved: {filename}")
                successful += 1
            
            # Save prompt info
            prompt_info = {
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "style": style,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(prompt_dir / "prompt_info.json", 'w', encoding='utf-8') as f:
                json.dump(prompt_info, f, indent=2, ensure_ascii=False)
            
            # Log
            utils.log_generation(prompt, True)
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failed += num_images_per_prompt
            utils.log_generation(prompt, False, str(e))
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*70)
    print("Batch Generation Complete")
    print("="*70)
    print(f"Total time: {utils.format_generation_time(total_time)}")
    print(f"Successful: {successful}/{total_images} images")
    print(f"Failed: {failed}/{total_images} images")
    print(f"Output directory: {output_dir}")
    print("="*70)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Batch generate images from multiple prompts'
    )
    
    parser.add_argument(
        'prompts',
        help='Path to file containing prompts (text or JSON) OR a single prompt string'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory (default: timestamped folder in generated_images/)',
        default=None
    )
    
    parser.add_argument(
        '--num-images', '-n',
        type=int,
        default=1,
        help='Number of images per prompt (default: 1)'
    )
    
    parser.add_argument(
        '--steps', '-s',
        type=int,
        default=30,
        help='Number of inference steps (default: 30)'
    )
    
    parser.add_argument(
        '--guidance', '-g',
        type=float,
        default=7.5,
        help='Guidance scale (default: 7.5)'
    )
    
    parser.add_argument(
        '--style',
        choices=list(config.STYLE_PRESETS.keys()),
        default='None',
        help='Style preset (default: None)'
    )
    
    parser.add_argument(
        '--no-negative-prompt',
        action='store_true',
        help='Disable negative prompt'
    )
    
    parser.add_argument(
        '--negative-prompt',
        help='Custom negative prompt'
    )
    
    args = parser.parse_args()
    
    # Load prompts
    try:
        prompts = load_prompts_from_file(args.prompts)
    except FileNotFoundError:
        # Treat as single prompt
        prompts = [args.prompts]
    
    # Run batch generation
    batch_generate(
        prompts=prompts,
        output_dir=args.output_dir,
        num_images_per_prompt=args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        style=args.style,
        use_negative_prompt=not args.no_negative_prompt,
        custom_negative_prompt=args.negative_prompt
    )

if __name__ == "__main__":
    main()
