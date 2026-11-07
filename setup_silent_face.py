#!/usr/bin/env python3
"""Setup script for Silent-Face-Anti-Spoofing MiniFASNet model."""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warning: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def setup_silent_face():
    """Verify and initialize Silent-Face-Anti-Spoofing MiniFASNet model."""
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    silent_face_dir = models_dir / "silent_face"
    
    print("Verifying Silent-Face-Anti-Spoofing for passive liveness detection...")
    print(f"Project root: {project_root}")
    print(f"Models directory: {models_dir}")
    
    # Create directories
    models_dir.mkdir(exist_ok=True)
    silent_face_dir.mkdir(exist_ok=True)
    
    # Verify model weights
    model_file = silent_face_dir / "2.7_80x80_MiniFASNetV2.pth"
    if not model_file.exists():
        print(f"\nModel weights file not found at: {model_file}")
        print("Please ensure '2.7_80x80_MiniFASNetV2.pth' is located in models/silent_face/.")
        return False
    else:
        print(f"\nModel weights confirmed: {model_file.name} ({model_file.stat().st_size / (1024*1024):.2f} MB)")
    
    # Verify model import and inference
    try:
        from liveness.silent_face_wrapper import get_silent_face_model
        model = get_silent_face_model()
        if model is not None and model.is_available:
            print("\nMiniFASNet neural model initialized and verified successfully!")
            print(f"Device: {model.device}, Input Resolution: {model.input_size}, Scale: {model.scale}")
        else:
            print("\nWarning: Model could not be loaded into memory.")
            return False
    except Exception as e:
        print(f"\nFailed to load MiniFASNet model: {e}")
        return False

    print("\nSetup & verification complete! Passive liveness PAD engine is ready.")
    return True

if __name__ == "__main__":
    success = setup_silent_face()
    sys.exit(0 if success else 1)
