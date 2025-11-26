"""
Example usage script demonstrating the API
This shows how to use the image generator programmatically
"""

from image_generator import ImageGenerator, PromptEnhancer
import utils
import config

def example_basic_generation():
    """Basic image generation example"""
    print("\n" + "="*60)
    print("Example 1: Basic Generation")
    print("="*60)
    
    # Create generator
    generator = ImageGenerator()
    
    # Load model
    print("Loading model...")
    generator.load_model()
    
    # Simple prompt
    prompt = "a serene mountain landscape with a lake"
    
    print(f"\nGenerating image with prompt: '{prompt}'")
    
    # Generate
    images, metadata = generator.generate(
        prompt=prompt,
        num_images=1,
        num_inference_steps=25,
        guidance_scale=7.5
    )
    
    # Save
    output_dir = utils.create_output_directory()
    for idx, img in enumerate(images):
        img_with_watermark = utils.add_watermark(img)
        filepath = f"{output_dir}/example_basic_{idx}.png"
        utils.save_image_with_metadata(
            img_with_watermark,
            prompt,
            filepath,
            metadata
        )
        print(f"Saved: {filepath}")
    
    print("✅ Basic generation complete!")

def example_styled_generation():
    """Generation with style presets"""
    print("\n" + "="*60)
    print("Example 2: Styled Generation")
    print("="*60)
    
    generator = ImageGenerator()
    generator.load_model()
    enhancer = PromptEnhancer()
    
    # Original prompt
    prompt = "a futuristic city"
    
    # Try different styles
    styles = ["Photorealistic", "Cyberpunk", "Oil Painting"]
    
    for style in styles:
        enhanced_prompt = enhancer.enhance_prompt(prompt, style)
        print(f"\nStyle: {style}")
        print(f"Enhanced prompt: {enhanced_prompt[:100]}...")
        
        images, metadata = generator.generate(
            prompt=enhanced_prompt,
            num_images=1,
            num_inference_steps=25
        )
        
        # Save
        output_dir = utils.create_output_directory()
        filepath = f"{output_dir}/example_style_{style.lower().replace(' ', '_')}.png"
        img_with_watermark = utils.add_watermark(images[0])
        utils.save_image_with_metadata(
            img_with_watermark,
            enhanced_prompt,
            filepath,
            metadata
        )
        print(f"Saved: {filepath}")
    
    print("\n✅ Styled generation complete!")

def example_batch_generation():
    """Generate multiple variations"""
    print("\n" + "="*60)
    print("Example 3: Batch Generation")
    print("="*60)
    
    generator = ImageGenerator()
    generator.load_model()
    
    prompt = "a magical forest with glowing mushrooms"
    
    print(f"\nGenerating 4 variations of: '{prompt}'")
    
    # Generate multiple images
    images, metadata = generator.generate(
        prompt=prompt,
        num_images=4,
        num_inference_steps=30,
        guidance_scale=8.0
    )
    
    # Save all
    output_dir = utils.create_output_directory()
    for idx, img in enumerate(images):
        img_with_watermark = utils.add_watermark(img)
        filepath = f"{output_dir}/example_batch_{idx+1}.png"
        utils.save_image_with_metadata(
            img_with_watermark,
            prompt,
            filepath,
            metadata
        )
        print(f"Saved variation {idx+1}: {filepath}")
    
    # Create grid
    grid = utils.create_image_grid(images, grid_size=(2, 2))
    grid_path = f"{output_dir}/example_batch_grid.png"
    grid.save(grid_path)
    print(f"Saved grid: {grid_path}")
    
    print("\n✅ Batch generation complete!")

def example_negative_prompts():
    """Using negative prompts for quality control"""
    print("\n" + "="*60)
    print("Example 4: Negative Prompts")
    print("="*60)
    
    generator = ImageGenerator()
    generator.load_model()
    
    prompt = "a portrait of a friendly robot"
    negative_prompt = "blurry, distorted, low quality, ugly, bad anatomy"
    
    print(f"\nPrompt: {prompt}")
    print(f"Negative: {negative_prompt}")
    
    images, metadata = generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images=1,
        num_inference_steps=35,
        guidance_scale=8.0
    )
    
    # Save
    output_dir = utils.create_output_directory()
    filepath = f"{output_dir}/example_negative.png"
    img_with_watermark = utils.add_watermark(images[0])
    utils.save_image_with_metadata(
        img_with_watermark,
        prompt,
        filepath,
        metadata
    )
    print(f"Saved: {filepath}")
    
    print("\n✅ Negative prompt generation complete!")

def example_reproducible_generation():
    """Generate with fixed seed for reproducibility"""
    print("\n" + "="*60)
    print("Example 5: Reproducible Generation")
    print("="*60)
    
    generator = ImageGenerator()
    generator.load_model()
    
    prompt = "a cute cat wearing sunglasses"
    seed = 42
    
    print(f"\nPrompt: {prompt}")
    print(f"Seed: {seed}")
    print("\nGenerating twice with same seed (should be identical)...")
    
    for run in [1, 2]:
        images, metadata = generator.generate(
            prompt=prompt,
            num_images=1,
            num_inference_steps=25,
            seed=seed
        )
        
        output_dir = utils.create_output_directory()
        filepath = f"{output_dir}/example_reproducible_run{run}.png"
        img_with_watermark = utils.add_watermark(images[0])
        utils.save_image_with_metadata(
            img_with_watermark,
            prompt,
            filepath,
            metadata
        )
        print(f"Run {run} saved: {filepath}")
    
    print("\n✅ Reproducible generation complete!")
    print("ℹ️  Compare the two images - they should be identical!")

def run_all_examples():
    """Run all examples"""
    print("\n" + "="*70)
    print("AI Image Generator - Example Usage")
    print("="*70)
    print("\nThis will demonstrate various features of the image generator.")
    print(f"Hardware: {config.DEVICE.upper()}")
    print(f"Model: {config.MODEL_ID}")
    
    if config.USE_GPU:
        print("⚡ Using GPU acceleration")
    else:
        print("⚠️  Using CPU (this will be slow - each example may take 5-10 minutes)")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    try:
        # Run examples
        example_basic_generation()
        example_styled_generation()
        example_batch_generation()
        example_negative_prompts()
        example_reproducible_generation()
        
        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70)
        print(f"\nGenerated images saved in: {config.OUTPUT_DIR}/")
        print("\nNext steps:")
        print("1. Review the generated images")
        print("2. Try the web interface: streamlit run app.py")
        print("3. Experiment with your own prompts!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~10GB)")
        print("3. Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    run_all_examples()
