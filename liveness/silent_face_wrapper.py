"""Wrapper for Silent-Face-Anti-Spoofing MiniFASNet model."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    return os.getenv("LIVENESS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_print(message: str) -> None:
    if _debug_enabled():
        print(message)


class SilentFaceWrapper:
    """Wrapper for Silent-Face-Anti-Spoofing MiniFASNet model."""
    
    def __init__(self, model_dir: Optional[str] = None, device_id: int = 0) -> None:
        """Initialize the wrapper.
        
        Args:
            model_dir: Directory containing the Silent-Face-Anti-Spoofing model files
            device_id: GPU device ID (0 for first GPU, -1 for CPU)
        """
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent.parent / "models" / "silent_face"
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu")
        self.model = None
        self._model_loaded = False
        
        # Try to load the model
        self._try_load_model()
    
    def _try_load_model(self) -> None:
        """Try to load the MiniFASNet model."""
        try:
            _debug_print(
                f"[DEBUG] SilentFaceWrapper: loading model. model_dir={self.model_dir}, device={self.device}"
            )
            # Ensure we're working with absolute paths
            if not self.model_dir.is_absolute():
                # Get the project root from this file's location
                project_root = Path(__file__).parent.parent.parent
                self.model_dir = project_root / self.model_dir
                _debug_print(f"[DEBUG] SilentFaceWrapper: resolved model_dir={self.model_dir}")
            
            # Check if we have the required model files
            model_file = self.model_dir / "2.7_80x80_MiniFASNetV2.pth"
            if not model_file.exists():
                logger.warning(f"MiniFASNet model not found at {model_file}")
                _debug_print(f"[DEBUG] SilentFaceWrapper: model file missing at {model_file}")
                return
            
            # Add the silent_face source to path if available
            silent_face_repo = self.model_dir / "Silent-Face-Anti-Spoofing"
            silent_face_src = silent_face_repo / "src"
            if not silent_face_src.exists():
                logger.warning(f"Silent-Face-Anti-Spoofing src directory not found at {silent_face_src}")
                _debug_print(f"[DEBUG] SilentFaceWrapper: src directory missing at {silent_face_src}")
                return
            
            # Temporarily add the repo root to path for imports (not src directory)
            # This allows "from src.xxx import yyy" imports to work
            import_path = str(silent_face_repo)
            if import_path not in sys.path:
                sys.path.insert(0, import_path)
            
            try:
                # Import the model
                from src.model_lib.MiniFASNet import MiniFASNetV2
                from src.data_io import transform as trans
                from src.utility import get_kernel, parse_model_name
                from src.generate_patches import CropImage
            finally:
                # Remove from path to avoid pollution
                if import_path in sys.path:
                    sys.path.remove(import_path)
            
            # Parse model info from filename
            model_name = model_file.name
            h_input, w_input, model_type, _ = parse_model_name(model_name)
            kernel_size = get_kernel(h_input, w_input)
            
            # Initialize model
            self.model = MiniFASNetV2(conv6_kernel=kernel_size).to(self.device)
            
            # Load weights
            state_dict = torch.load(str(model_file), map_location=self.device)
            keys = iter(state_dict)
            first_layer_name = next(keys)
            if first_layer_name.find('module.') >= 0:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    name_key = key[7:]
                    new_state_dict[name_key] = value
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)
            
            self.model.eval()
            self._model_loaded = True
            self.transform = trans.Compose([trans.ToTensor()])
            self.crop_image = CropImage()
            self.input_size = (w_input, h_input)  # Store expected input dimensions
            
            logger.info(f"MiniFASNet model loaded successfully from {model_file}")
            logger.info(f"Model expects input size: {w_input}x{h_input}")
            _debug_print(
                f"[DEBUG] SilentFaceWrapper: model loaded, input_size={w_input}x{h_input}"
            )
            
        except ImportError as e:
            logger.warning(f"Could not import Silent-Face-Anti-Spoofing modules: {e}")
            _debug_print(f"[DEBUG] SilentFaceWrapper: import failed: {e}")
        except Exception as e:
            logger.error(f"Failed to load MiniFASNet model: {e}")
            _debug_print(f"[DEBUG] SilentFaceWrapper: load failed: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self._model_loaded and self.model is not None
    
    def predict(self, face_image: np.ndarray) -> np.ndarray:
        """Predict liveness score for a face image.
        
        Args:
            face_image: Input face image as numpy array (BGR or RGB)
            
        Returns:
            numpy array with liveness scores [spoof_prob, real_prob]
        """
        if not self.is_available:
            raise RuntimeError("MiniFASNet model is not available")
        
        # Ensure image is in the right format (uint8 for cv2 operations)
        if face_image.dtype != np.uint8:
            if face_image.max() <= 1.0:
                face_image = (face_image * 255).astype(np.uint8)
            else:
                face_image = face_image.astype(np.uint8)
        
        # Resize image to model's expected input dimensions
        # The model expects a specific input size (e.g., 80x80)
        w_input, h_input = self.input_size
        resized_image = cv2.resize(face_image, (w_input, h_input))
        
        # Apply transforms
        img_tensor = self.transform(resized_image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            result = self.model.forward(img_tensor)
            result = F.softmax(result).cpu().numpy()
        
        return result[0]  # Return [spoof_prob, real_prob]


# Create a singleton instance for easier import
_silent_face_instance: Optional[SilentFaceWrapper] = None


def get_silent_face_model(model_dir: Optional[str] = None, device_id: int = 0) -> Optional[SilentFaceWrapper]:
    """Get or create a SilentFaceWrapper instance."""
    global _silent_face_instance
    if _silent_face_instance is None:
        _silent_face_instance = SilentFaceWrapper(model_dir, device_id)
    return _silent_face_instance if _silent_face_instance.is_available else None


# For compatibility with the existing passive.py import
class MiniFASNet:
    """Compatibility class that mimics the expected MiniFASNet interface."""
    
    def __init__(self, model_dir: Optional[str] = None, device_id: int = 0) -> None:
        self.wrapper = get_silent_face_model(model_dir, device_id)
        if self.wrapper is None:
            raise RuntimeError("MiniFASNet model is not available")
    
    def predict(self, face_image: np.ndarray) -> np.ndarray:
        """Predict liveness score."""
        if self.wrapper is None:
            raise RuntimeError("MiniFASNet model is not available")
        return self.wrapper.predict(face_image)
