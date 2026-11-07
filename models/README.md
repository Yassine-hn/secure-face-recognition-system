# Model Weights Setup

This directory contains model weights for the face recognition system.

## MiniFASNet for Passive Liveness (PAD)

Passive liveness detection uses the **MiniFASNetV2** architecture. The model architecture is natively implemented in `liveness/minifasnet_model.py` and the pre-trained neural network weights are included directly in this repository:

- **Weight file**: `models/silent_face/2.7_80x80_MiniFASNetV2.pth` (1.8 MB)
- **Input resolution**: 80x80 with scale 2.7 contextual face crop
- **Dependencies**: Native PyTorch & OpenCV (zero external repository cloning needed)

### Verify Setup

Run this command to verify that the neural model is available:
```bash
python -c "from liveness.silent_face_wrapper import get_silent_face_model; print('Model available:' if get_silent_face_model() else 'Model not available')"
```

## InsightFace Models (RetinaFace & ArcFace)

InsightFace models (`buffalo_l` suite for face detection and 512-D ArcFace embedding extraction) are loaded via `FaceAnalysis` and cached automatically on first run to:
- `~/.insightface/models/buffalo_l/`

