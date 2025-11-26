"""
Core Image Generation Module
Handles text-to-image generation using Stable Diffusion
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import config
import os
from typing import List, Optional, Tuple
import gc

class ImageGenerator:
    """
    Text-to-image generator using Stable Diffusion
    """
    
    def __init__(self):
        """Initialize the image generator"""
        self.pipe = None
        self.device = config.DEVICE
        self.is_loaded = False
        
    def load_model(self, progress_callback=None):
        """
        Load the Stable Diffusion model
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            bool: Success status
        """
        try:
            if progress_callback:
                progress_callback("Loading Stable Diffusion model...")
            
            # Create cache directory if it doesn't exist
            os.makedirs(config.MODEL_CACHE_DIR, exist_ok=True)
            
            # Load the model
            if progress_callback:
                progress_callback("Downloading model files (first time only, ~4-5GB)...")
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                config.MODEL_ID,
                torch_dtype=torch.float16 if config.USE_GPU else torch.float32,
                cache_dir=config.MODEL_CACHE_DIR,
                safety_checker=None,  # We implement custom content filtering
                requires_safety_checker=False
            )
            
            # Use DPM Solver for faster generation
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Apply optimizations
            if config.ENABLE_ATTENTION_SLICING:
                self.pipe.enable_attention_slicing()
            
            if config.ENABLE_VAE_SLICING:
                self.pipe.enable_vae_slicing()
            
            if config.ENABLE_CPU_OFFLOAD and config.USE_GPU:
                self.pipe.enable_model_cpu_offload()
            
            self.is_loaded = True
            
            if progress_callback:
                progress_callback("Model loaded successfully!")
            
            return True
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            if progress_callback:
                progress_callback(error_msg)
            print(error_msg)
            return False
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        width: int = None,
        height: int = None,
        seed: Optional[int] = None,
        progress_callback=None
    ) -> Tuple[List, dict]:
        """
        Generate images from text prompt
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            num_images: Number of images to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            width: Image width
            height: Image height
            seed: Random seed for reproducibility
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (list of PIL Images, generation metadata dict)
        """
        if not self.is_loaded:
            if not self.load_model(progress_callback):
                raise RuntimeError("Failed to load model")
        
        # Set defaults
        if num_inference_steps is None:
            num_inference_steps = config.DEFAULT_NUM_INFERENCE_STEPS
        if guidance_scale is None:
            guidance_scale = config.DEFAULT_GUIDANCE_SCALE
        if width is None:
            width = config.DEFAULT_IMAGE_SIZE
        if height is None:
            height = config.DEFAULT_IMAGE_SIZE
        if negative_prompt is None:
            negative_prompt = config.DEFAULT_NEGATIVE_PROMPT
        
        try:
            if progress_callback:
                progress_callback(f"Generating {num_images} image(s)...")
            
            # Set random seed if provided
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate images
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                )
            
            images = result.images
            
            # Create metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_images": num_images,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed,
                "model": config.MODEL_ID,
                "device": self.device
            }
            
            if progress_callback:
                progress_callback("Generation complete!")
            
            return images, metadata
            
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            if progress_callback:
                progress_callback(error_msg)
            raise RuntimeError(error_msg)
    
    def unload_model(self):
        """
        Unload model from memory to free up resources
        """
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.is_loaded = False
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Garbage collection
            gc.collect()
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_id": config.MODEL_ID,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "use_gpu": config.USE_GPU,
        }
        
        if config.USE_GPU and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
            if self.is_loaded:
                info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / (1024**3):.1f} GB"
                info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved(0) / (1024**3):.1f} GB"
        
        return info


class PromptEnhancer:
    """
    Helper class for enhancing prompts with quality modifiers
    """
    
    @staticmethod
    def enhance_prompt(prompt: str, style: str = "None") -> str:
        """
        Enhance prompt with style modifiers
        
        Args:
            prompt: Original prompt
            style: Style preset name
            
        Returns:
            Enhanced prompt
        """
        style_modifier = config.STYLE_PRESETS.get(style, "")
        enhanced = prompt + style_modifier
        return enhanced.strip()
    
    @staticmethod
    def add_quality_tags(prompt: str) -> str:
        """
        Add quality enhancement tags if not already present
        
        Args:
            prompt: Original prompt
            
        Returns:
            Prompt with quality tags
        """
        quality_tags = ["highly detailed", "high quality", "4K", "8K", "detailed"]
        prompt_lower = prompt.lower()
        
        # Check if prompt already has quality tags
        has_quality_tag = any(tag in prompt_lower for tag in quality_tags)
        
        if not has_quality_tag:
            prompt = prompt + ", highly detailed, high quality"
        
        return prompt
    
    @staticmethod
    def get_style_suggestions() -> List[str]:
        """
        Get list of available style presets
        
        Returns:
            List of style names
        """
        return list(config.STYLE_PRESETS.keys())


# Convenience function for quick generation
def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    style: str = "None",
    num_images: int = 1,
    quality: str = "balanced"
) -> List:
    """
    Quick generation function with simplified parameters
    
    Args:
        prompt: Text description
        negative_prompt: What to avoid
        style: Style preset
        num_images: Number of images
        quality: Quality preset ("fast", "balanced", "high")
        
    Returns:
        List of generated PIL Images
    """
    # Quality presets
    quality_settings = {
        "fast": {"steps": 20, "guidance": 7.0},
        "balanced": {"steps": 30, "guidance": 7.5},
        "high": {"steps": 50, "guidance": 8.0}
    }
    
    settings = quality_settings.get(quality, quality_settings["balanced"])
    
    # Enhance prompt
    enhancer = PromptEnhancer()
    enhanced_prompt = enhancer.enhance_prompt(prompt, style)
    
    # Generate
    generator = ImageGenerator()
    images, _ = generator.generate(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        num_images=num_images,
        num_inference_steps=settings["steps"],
        guidance_scale=settings["guidance"]
    )
    
    return images
