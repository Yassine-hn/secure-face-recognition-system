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
    """Download and set up Silent-Face-Anti-Spoofing."""
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    silent_face_dir = models_dir / "silent_face"
    
    print("Setting up Silent-Face-Anti-Spoofing for passive liveness detection...")
    print(f"Project root: {project_root}")
    print(f"Models directory: {models_dir}")
    
    # Create directories
    models_dir.mkdir(exist_ok=True)
    silent_face_dir.mkdir(exist_ok=True)
    
    # Clone the repository if not already present
    repo_dir = silent_face_dir / "Silent-Face-Anti-Spoofing"
    if not repo_dir.exists():
        print("\n1. Cloning Silent-Face-Anti-Spoofing repository...")
        # Use subprocess.run with list to avoid shell escaping issues
        import subprocess
        try:
            subprocess.run(["git", "clone", "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git", str(repo_dir)], 
                         check=True, cwd=silent_face_dir)
            print("Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
            return False
    else:
        print("\n1. Repository already exists, skipping clone...")
    
    # Download model weights
    model_file = silent_face_dir / "2.7_80x80_MiniFASNetV2.pth"
    if not model_file.exists():
        print("\n2. Downloading MiniFASNet model weights...")
        # The model needs to be downloaded from a specific location
        # For now, we'll provide instructions since direct download URLs may vary
        print("Please download the MiniFASNet model manually:")
        print("1. Visit: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/releases")
        print("2. Download the '2.7_80x80_MiniFASNetV2.pth' model file")
        print(f"3. Place it in: {model_file}")
        print("\nAlternatively, you can find pre-trained models in the repository's resources folder.")
        return False
    else:
        print("\n2. Model weights already exist, skipping download...")
    
    print("\n3. Setup complete!")
    print(f"Silent-Face-Anti-Spoofing is ready at: {silent_face_dir}")
    print("\nTo verify the setup, run:")
    print("python -c \"from liveness.silent_face_wrapper import get_silent_face_model; print('Model available:' if get_silent_face_model() else 'Model not available')\"")
    
    return True

if __name__ == "__main__":
    success = setup_silent_face()
    sys.exit(0 if success else 1)
