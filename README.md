# 🎨 AI-Powered Image Generator

> Transform Your Ideas into Stunning Visual Art with Artificial Intelligence

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-v1.5-purple.svg)](https://huggingface.co/runwayml/stable-diffusion-v1-5)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A sophisticated text-to-image generation system powered by **Stable Diffusion** that converts natural language descriptions into high-quality, professional-grade images.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎨 **Text-to-Image** | Generate images from text descriptions using Stable Diffusion v1.5 |
| 🖥️ **Web Interface** | Beautiful Streamlit UI with real-time progress tracking |
| ⚡ **GPU & CPU Support** | CUDA acceleration for NVIDIA GPUs, CPU fallback for all systems |
| 🎭 **13 Style Presets** | Photorealistic, Anime, Oil Painting, Cyberpunk, and more |
| 📁 **Smart Storage** | Automatic organization with metadata and timestamps |
| 🛡️ **Content Safety** | Built-in content filtering and AI watermarking |
| 🔧 **Customizable** | Adjustable steps, guidance scale, seeds, and resolutions |
| 📦 **Batch Processing** | Generate multiple images with CLI tools |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 - 3.11
- 8GB+ RAM (16GB recommended)
- 10GB free disk space
- NVIDIA GPU with 6GB+ VRAM (optional, for faster generation)

### Installation

```powershell
# 1. Clone or download the project
git clone https://github.com/ayusingh-54/AI-Powered-Image-Generator.git
cd AI-Powered-Image-Generator

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements-cpu.txt     # For CPU
# OR
pip install -r requirements.txt         # For GPU (NVIDIA)

# 5. Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
ai-image-generator/
├── app.py                 # Main Streamlit web application
├── image_generator.py     # Core AI generation engine
├── config.py              # Configuration and settings
├── utils.py               # Helper functions
├── batch_generate.py      # CLI batch processing
├── examples.py            # Usage examples
├── test_setup.py          # Installation verification
├── requirements.txt       # GPU dependencies
├── requirements-cpu.txt   # CPU dependencies
├── setup.ps1              # Automated setup script
├── run.ps1                # Quick launch script
├── GETTING_STARTED.md     # Beginner's guide
├── ETHICAL_GUIDELINES.md  # Responsible AI usage
└── PROJECT_STRUCTURE.md   # Technical architecture
```

---

## 🎮 Usage

### Web Interface

1. **Enter your prompt** - Describe the image you want to create
2. **Choose settings** - Adjust steps, guidance scale, size, and style
3. **Click Generate** - Wait for AI to create your image
4. **Download** - Save individual images or batch download as ZIP

### Example Prompts

```
"A serene mountain lake at sunset, professional photography, 4K"
"A cyberpunk city with neon lights, rain, detailed, futuristic"
"Portrait of a robot in Van Gogh style, oil painting, artistic"
"A magical forest with glowing mushrooms, fantasy art, ethereal"
```

### Command Line

```powershell
# Batch generate from file
python batch_generate.py --prompts example_prompts.txt --output ./batch_output
```

---

## ⚙️ Configuration

### Style Presets

| Style | Description |
|-------|-------------|
| Photorealistic | Professional photography look |
| Digital Art | Modern digital illustration |
| Oil Painting | Classical fine art style |
| Watercolor | Soft, flowing paint effect |
| Anime | Japanese animation style |
| Cyberpunk | Futuristic neon aesthetic |
| Fantasy | Magical, mythical artwork |
| 3D Render | CGI quality graphics |
| Sketch | Hand-drawn pencil look |
| Vintage | Retro, aged photograph |

### Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Steps | 10-100 | 30 | More steps = better quality, slower |
| Guidance | 1-20 | 7.5 | Higher = follows prompt more closely |
| Size | 512-768 | 512x512 | Image resolution |
| Seed | Any | Random | For reproducible results |

---

## 💻 System Requirements

### Minimum (CPU Mode)

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 10.15+ |
| **CPU** | Intel i5 / AMD Ryzen 5 |
| **RAM** | 8GB |
| **Storage** | 10GB free |
| **Speed** | ~5-10 min per image |

### Recommended (GPU Mode)

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11, Linux |
| **GPU** | NVIDIA RTX 3060+ (8GB+ VRAM) |
| **CPU** | Intel i7 / AMD Ryzen 7 |
| **RAM** | 16GB |
| **Storage** | 20GB SSD |
| **Speed** | ~5-10 sec per image |

---

## 🛡️ Ethical Guidelines

This tool includes built-in safety features:

- ✅ **Content Filtering** - Blocks inappropriate prompts
- ✅ **AI Watermarking** - All images marked as AI-generated
- ✅ **Usage Logging** - Audit trail for accountability
- ✅ **Metadata Tracking** - Full generation parameters saved

### Responsible Use

| ✔️ Allowed | ❌ Prohibited |
|-----------|--------------|
| Creative art and design | Deepfakes or misinformation |
| Educational purposes | Illegal content |
| Personal projects | Copyright infringement |
| Concept prototyping | Impersonation |

See [ETHICAL_GUIDELINES.md](ETHICAL_GUIDELINES.md) for complete details.

---

## 🔧 Troubleshooting

### CUDA Out of Memory

```python
# Reduce image size to 512x512
# Lower batch size to 1
# Or use CPU mode
```

### Module Not Found Error

```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements-cpu.txt --force-reinstall
```

### Slow Generation

- CPU mode takes 5-10 minutes per image (normal)
- Reduce inference steps to 20-25
- Use 512x512 resolution
- Close other applications

### Virtual Environment Won't Activate

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

---

## 📖 API Reference

### ImageGenerator Class

```python
from image_generator import ImageGenerator

# Initialize
generator = ImageGenerator()

# Load model (downloads ~5GB on first run)
generator.load_model()

# Generate images
images = generator.generate(
    prompt="a beautiful sunset",
    negative_prompt="blurry, low quality",
    num_images=1,
    num_inference_steps=30,
    guidance_scale=7.5,
    width=512,
    height=512,
    seed=42  # Optional, for reproducibility
)

# Save result
images[0].save("output.png")

# Free memory
generator.unload_model()
```

### PromptEnhancer Class

```python
from image_generator import PromptEnhancer

enhancer = PromptEnhancer()
enhanced = enhancer.enhance_prompt("a cat", style="Photorealistic")
# Result: "a cat, photorealistic, highly detailed, 8k uhd..."
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Streamlit Web UI              │
│      (app.py - User Interface)          │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         Application Layer               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Config  │ │ Utils   │ │ Filter  │   │
│  └─────────┘ └─────────┘ └─────────┘   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│       AI Generation Engine              │
│     (image_generator.py)                │
│  ┌─────────────────────────────────┐   │
│  │    Stable Diffusion v1.5        │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐    │   │
│  │  │ CLIP │ │ UNet │ │ VAE  │    │   │
│  │  └──────┘ └──────┘ └──────┘    │   │
│  └─────────────────────────────────┘   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         PyTorch Backend                 │
│      (CUDA GPU / CPU Fallback)          │
└─────────────────────────────────────────┘
```

---

## 📊 Performance Benchmarks

| Configuration | Resolution | Steps | Time |
|---------------|------------|-------|------|
| RTX 4090 | 512x512 | 30 | ~3s |
| RTX 3060 | 512x512 | 30 | ~8s |
| RTX 3060 | 768x768 | 30 | ~15s |
| CPU (i7) | 512x512 | 30 | ~5min |
| CPU (i7) | 512x512 | 20 | ~3.5min |

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

- **Code**: MIT License
- **Model**: [CreativeML Open RAIL-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

---

## 🙏 Acknowledgments

- **[Stability AI](https://stability.ai/)** - Stable Diffusion model
- **[Hugging Face](https://huggingface.co/)** - Diffusers library
- **[Streamlit](https://streamlit.io/)** - Web framework
- **[PyTorch](https://pytorch.org/)** - Deep learning framework

---

## 📞 Support

- 📚 [Getting Started Guide](GETTING_STARTED.md)
- 🐛 [Report Issues](https://github.com/ayusingh-54/AI-Powered-Image-Generator/issues)
- 💬 [Discussions](https://github.com/ayusingh-54/AI-Powered-Image-Generator/discussions)

---

<div align="center">

**Built with ❤️ for the AI community**

*Part of the Talrn.com AI Selection Task*

⭐ Star this repo if you find it helpful!

</div>

