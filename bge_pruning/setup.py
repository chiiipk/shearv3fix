#!/usr/bin/env python3
"""
One-command setup for BGE-M3 Pruning
Clean installation with dependency verification
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required packages"""
    print("🔧 Installing BGE-M3 Pruning dependencies...")
    
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements_clean.txt"]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    
    return True

def verify_installation():
    """Verify critical imports"""
    print("🔍 Verifying installation...")
    
    try:
        import torch
        import transformers
        import datasets
        from composer import Trainer
        print("✅ All imports verified!")
        return True
    except ImportError as e:
        print(f"❌ Import verification failed: {e}")
        return False

def main():
    print("🚀 BGE-M3 Pruning Setup")
    print("=" * 30)
    
    if not install_dependencies():
        sys.exit(1)
    
    if not verify_installation():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n🔥 Ready to run:")
    print("   python train_clean.py config_clean.yaml --dataset sts")
    print("   python train_clean.py config_clean.yaml --dataset msmarco")

if __name__ == "__main__":
    main()
