# Model Weights Setup

This directory contains model weights for the face recognition system.

## MiniFASNet for Passive Liveness

The passive liveness detection uses MiniFASNet from Silent-Face-Anti-Spoofing. Since this is not a proper Python package, it needs manual setup:

### Option 1: Automated Setup (Recommended)

```bash
python setup_silent_face.py
```

This script will:
1. Clone the Silent-Face-Anti-Spoofing repository to `models/silent_face/`
2. Provide instructions for downloading the model weights

### Option 2: Manual Setup

1. Clone the repository:
```bash
cd models
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git silent_face/Silent-Face-Anti-Spoofing
```

2. Download the model weights:
   - Visit: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/releases
   - Download: `2.7_80x80_MiniFASNetV2.pth`
   - Place it in: `models/silent_face/2.7_80x80_MiniFASNetV2.pth`

### Verify Setup

Run this command to verify the model is available:
```bash
python -c "from liveness.silent_face_wrapper import get_silent_face_model; print('Model available:' if get_silent_face_model() else 'Model not available')"
```

## InsightFace Models

InsightFace models (RetinaFace and ArcFace) are downloaded automatically on first use to:
- `~/.insightface/models/buffalo_l/`
