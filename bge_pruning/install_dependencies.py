#!/usr/bin/env python3
"""
Elegant dependency installer for BGE-M3 Pruning Project
Installs only essential dependencies with proper version handling
"""

import subprocess
import sys
import importlib

CORE_DEPS = [
    "torch>=2.0.0",
    "transformers>=4.30.0", 
    "mosaicml[all]>=0.17.0",
    "omegaconf>=2.3.0",
    "wandb>=0.15.0",
    "scipy>=1.9.0",  # For Spearman correlation fallback
]

def check_dependency(package_name):
    """Check if package is already installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install dependencies efficiently"""
    print("🔍 Checking BGE-M3 Pruning dependencies...")
    
    # Check which deps are missing
    missing_deps = []
    for dep in CORE_DEPS:
        pkg_name = dep.split(">=")[0].split("[")[0]
        if not check_dependency(pkg_name):
            missing_deps.append(dep)
    
    if not missing_deps:
        print("✅ All dependencies already installed!")
        return
    
    print(f"📦 Installing {len(missing_deps)} missing dependencies...")
    
    # Install missing deps
    cmd = [sys.executable, "-m", "pip", "install"] + missing_deps
    try:
        subprocess.run(cmd, check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)

def verify_installation():
    """Verify critical imports work"""
    try:
        import torch
        import transformers
        from composer.models.base import ComposerModel
        print("✅ Core imports verified!")
    except ImportError as e:
        print(f"❌ Import verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_dependencies()
    verify_installation()
    print("🚀 BGE-M3 Pruning project ready!")
