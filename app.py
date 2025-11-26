"""
AI Image Generator - Streamlit Web Interface
Main application file
"""

import streamlit as st
import torch
from PIL import Image
import io
import time
import os
from datetime import datetime
import zipfile

# Import custom modules
import config
import utils
from image_generator import ImageGenerator, PromptEnhancer

# Page configuration
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = ImageGenerator()
    st.session_state.generated_images = []
    st.session_state.generation_metadata = []
    st.session_state.model_loaded = False

def load_model_cached():
    """Load model if not already loaded"""
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading AI model (first time only)..."):
            progress_placeholder = st.empty()
            
            def progress_callback(msg):
                progress_placeholder.info(msg)
            
            success = st.session_state.generator.load_model(progress_callback)
            
            if success:
                st.session_state.model_loaded = True
                progress_placeholder.success("✅ Model loaded successfully!")
                time.sleep(1)
                progress_placeholder.empty()
            else:
                progress_placeholder.error("❌ Failed to load model. Please check your setup.")
                st.stop()

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🎨 AI Image Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transform your ideas into stunning images with AI</div>', unsafe_allow_html=True)
    
    # Hardware info
    st.info(config.HARDWARE_INFO)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model info
        with st.expander("📊 Model Information"):
            st.write(f"**Model:** {config.MODEL_ID}")
            st.write(f"**Device:** {config.DEVICE.upper()}")
            if config.USE_GPU and torch.cuda.is_available():
                st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
        
        st.divider()
        
        # Generation parameters
        st.subheader("🎛️ Generation Parameters")
        
        num_images = st.slider(
            "Number of Images",
            min_value=1,
            max_value=4,
            value=1,
            help="More images = longer generation time"
        )
        
        num_steps = st.slider(
            "Inference Steps",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="More steps = better quality but slower. 20-50 recommended."
        )
        
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=7.5,
            step=0.5,
            help="How closely to follow the prompt. 7-15 recommended."
        )
        
        image_size = st.select_slider(
            "Image Size",
            options=list(config.IMAGE_SIZE_OPTIONS.keys()),
            value="Small (512x512)",
            help="Larger images require more VRAM and time"
        )
        size_value = config.IMAGE_SIZE_OPTIONS[image_size]
        
        style_preset = st.selectbox(
            "Style Preset",
            options=list(config.STYLE_PRESETS.keys()),
            help="Apply a style to your generation"
        )
        
        use_seed = st.checkbox("Use Custom Seed (for reproducibility)")
        seed = None
        if use_seed:
            seed = st.number_input("Seed", min_value=0, max_value=999999, value=42)
        
        st.divider()
        
        # Estimated time
        est_time = utils.estimate_generation_time(num_images, num_steps)
        st.info(f"⏱️ Estimated time: {utils.format_generation_time(est_time)}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Prompt Input")
        
        # Example prompts
        example_prompts = utils.get_prompt_suggestions()
        selected_example = st.selectbox(
            "💡 Example Prompts (optional)",
            options=["Custom prompt..."] + example_prompts,
            index=0
        )
        
        # Text prompt
        default_prompt = "" if selected_example == "Custom prompt..." else selected_example
        prompt = st.text_area(
            "Describe the image you want to generate:",
            value=default_prompt,
            height=100,
            placeholder="e.g., a futuristic city at sunset, highly detailed, 4K",
            help="Be specific and descriptive for best results"
        )
        
        # Negative prompt
        use_negative = st.checkbox("Use Negative Prompt", value=True)
        negative_prompt = ""
        if use_negative:
            negative_prompt = st.text_area(
                "What to avoid in the image:",
                value=config.DEFAULT_NEGATIVE_PROMPT,
                height=80,
                help="Specify unwanted elements"
            )
        
        # Export format
        export_format = st.selectbox(
            "Export Format",
            options=config.SUPPORTED_FORMATS,
            index=0
        )
        
        # Generate button
        generate_button = st.button("🎨 Generate Images", type="primary")
        
        # Tips
        with st.expander("💡 Tips for Better Results"):
            st.markdown("""
            - **Be specific**: Instead of "a cat", try "a fluffy orange tabby cat sitting on a windowsill"
            - **Add quality tags**: "highly detailed", "4K", "professional photography"
            - **Specify style**: "oil painting", "digital art", "photorealistic"
            - **Use negative prompts**: Exclude unwanted elements like "blurry", "distorted"
            - **Experiment**: Try different guidance scales and steps
            """)
    
    with col2:
        st.subheader("🖼️ Generated Images")
        
        # Generation logic
        if generate_button:
            if not prompt.strip():
                st.error("⚠️ Please enter a prompt first!")
            else:
                # Content filtering
                is_safe, filtered_prompt, warning = utils.filter_inappropriate_content(prompt)
                
                if not is_safe:
                    st.error(warning)
                    st.warning("Please modify your prompt and try again.")
                    utils.log_generation(prompt, False, "Content filter triggered")
                else:
                    # Load model
                    load_model_cached()
                    
                    # Enhance prompt with style
                    enhancer = PromptEnhancer()
                    enhanced_prompt = enhancer.enhance_prompt(prompt, style_preset)
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        start_time = time.time()
                        
                        status_text.text("🎨 Generating your images...")
                        progress_bar.progress(30)
                        
                        # Generate images
                        images, metadata = st.session_state.generator.generate(
                            prompt=enhanced_prompt,
                            negative_prompt=negative_prompt if use_negative else None,
                            num_images=num_images,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance_scale,
                            width=size_value,
                            height=size_value,
                            seed=seed
                        )
                        
                        progress_bar.progress(80)
                        
                        # Add watermarks
                        watermarked_images = []
                        for img in images:
                            watermarked = utils.add_watermark(img)
                            watermarked_images.append(watermarked)
                        
                        progress_bar.progress(90)
                        
                        # Save images
                        output_dir = utils.create_output_directory()
                        saved_paths = []
                        
                        for idx, img in enumerate(watermarked_images):
                            filename = f"image_{idx+1}.{export_format.lower()}"
                            filepath = os.path.join(output_dir, filename)
                            
                            # Convert format if needed
                            img_to_save = utils.convert_image_format(img, export_format)
                            
                            # Save with metadata
                            utils.save_image_with_metadata(
                                img_to_save,
                                enhanced_prompt,
                                filepath,
                                utils.create_metadata_dict(enhanced_prompt, negative_prompt, metadata)
                            )
                            saved_paths.append(filepath)
                        
                        progress_bar.progress(100)
                        
                        # Calculate generation time
                        end_time = time.time()
                        generation_time = end_time - start_time
                        
                        # Store results
                        st.session_state.generated_images = watermarked_images
                        st.session_state.generation_metadata = {
                            "metadata": metadata,
                            "saved_paths": saved_paths,
                            "generation_time": generation_time,
                            "original_prompt": prompt,
                            "enhanced_prompt": enhanced_prompt
                        }
                        
                        # Success message
                        status_text.success(
                            f"✅ Generated {len(images)} image(s) in {utils.format_generation_time(generation_time)}"
                        )
                        progress_bar.empty()
                        
                        # Log success
                        utils.log_generation(prompt, True)
                        
                    except Exception as e:
                        progress_bar.empty()
                        status_text.error(f"❌ Generation failed: {str(e)}")
                        utils.log_generation(prompt, False, str(e))
        
        # Display generated images
        if st.session_state.generated_images:
            st.success(f"✨ Successfully generated {len(st.session_state.generated_images)} image(s)!")
            
            # Display metadata
            if st.session_state.generation_metadata:
                meta = st.session_state.generation_metadata
                with st.expander("📋 Generation Details"):
                    st.write(f"**Original Prompt:** {meta['original_prompt']}")
                    st.write(f"**Enhanced Prompt:** {meta['enhanced_prompt']}")
                    st.write(f"**Generation Time:** {utils.format_generation_time(meta['generation_time'])}")
                    st.write(f"**Saved to:** {os.path.dirname(meta['saved_paths'][0])}")
            
            # Display images in grid
            cols_per_row = 2
            for i in range(0, len(st.session_state.generated_images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(st.session_state.generated_images):
                        with cols[j]:
                            img = st.session_state.generated_images[idx]
                            st.image(img, use_container_width=True)
                            
                            # Download button
                            buf = io.BytesIO()
                            img_to_download = utils.convert_image_format(img, export_format)
                            img_to_download.save(buf, format=export_format)
                            buf.seek(0)
                            
                            st.download_button(
                                label=f"⬇️ Download Image {idx+1}",
                                data=buf,
                                file_name=f"ai_generated_{idx+1}.{export_format.lower()}",
                                mime=f"image/{export_format.lower()}"
                            )
            
            # Download all as ZIP
            if len(st.session_state.generated_images) > 1:
                st.divider()
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, img in enumerate(st.session_state.generated_images):
                        img_buffer = io.BytesIO()
                        img_to_save = utils.convert_image_format(img, export_format)
                        img_to_save.save(img_buffer, format=export_format)
                        img_buffer.seek(0)
                        zip_file.writestr(
                            f"ai_generated_{idx+1}.{export_format.lower()}",
                            img_buffer.getvalue()
                        )
                
                zip_buffer.seek(0)
                st.download_button(
                    label="📦 Download All Images (ZIP)",
                    data=zip_buffer,
                    file_name=f"ai_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
    
    # Footer
    st.divider()
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("### 📚 About")
        st.markdown("AI-powered image generation using Stable Diffusion")
    
    with col_b:
        st.markdown("### ⚖️ Ethical Use")
        with st.expander("View Guidelines"):
            st.markdown("""
            - Use responsibly and ethically
            - Respect copyright and intellectual property
            - No harmful, explicit, or discriminatory content
            - AI-generated images are watermarked
            - See ETHICAL_GUIDELINES.md for details
            """)
    
    with col_c:
        st.markdown("### 🔧 Model Info")
        st.markdown(f"**Model:** Stable Diffusion 1.5")
        st.markdown(f"**Mode:** {'GPU' if config.USE_GPU else 'CPU'}")

if __name__ == "__main__":
    main()
