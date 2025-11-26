"""
Test script to verify installation and basic functionality
Run this after setup to ensure everything is working correctly
"""

import sys
import importlib

def check_python_version():
    """Check if Python version is adequate"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Too old!")
        print("   Required: Python 3.8 or higher")
        return False

def check_package(package_name, display_name=None):
    """Check if a package is installed"""
    if display_name is None:
        display_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✅ {display_name} - Installed")
        return True
    except ImportError:
        print(f"❌ {display_name} - Not installed")
        return False

def check_torch_gpu():
    """Check if PyTorch can access GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU Available: {gpu_name}")
            return True
        else:
            print("ℹ️  GPU: Not available (will use CPU)")
            return True
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
        return False

def check_disk_space():
    """Check available disk space"""
    import shutil
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        print(f"ℹ️  Free disk space: {free_gb} GB")
        if free_gb < 10:
            print("⚠️  Warning: Less than 10GB free. Model download may fail.")
            return False
        else:
            print("✅ Sufficient disk space")
            return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True

def main():
    """Run all checks"""
    print("=" * 60)
    print("AI Image Generator - Installation Check")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    if not check_python_version():
        all_ok = False
    print()
    
    # Check required packages
    print("Checking required packages...")
    packages = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("streamlit", "Streamlit"),
        ("numpy", "NumPy"),
    ]
    
    for package, display_name in packages:
        if not check_package(package, display_name):
            all_ok = False
    print()
    
    # Check GPU
    print("Checking GPU availability...")
    check_torch_gpu()
    print()
    
    # Check disk space
    print("Checking disk space...")
    if not check_disk_space():
        all_ok = False
    print()
    
    # Final summary
    print("=" * 60)
    if all_ok:
        print("✅ All checks passed! You're ready to generate images.")
        print()
        print("To start the application, run:")
        print("   streamlit run app.py")
    else:
        print("❌ Some checks failed. Please install missing packages:")
        print("   pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()
