# secure-face-recognition-system
This project develops a secure student face recognition system using deep learning techniques. It can be used to track student attendance or perform authentication. The system is designed to provide a level of security that prevents malicious attempts, such as using photos or crafted masks, from bypassing recognition.


---
 
## Table of Contents
 
- [System Overview](#system-overview)
- [Pipeline & System Layers](#pipeline--system-layers)
- [Project Structure](#project-structure)
- [Tools, Libraries, Frameworks & Models](#tools-libraries-frameworks--models)
- [Security Measures](#security-measures)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [What Is Intentionally Excluded from the Repository](#what-is-intentionally-excluded-from-the-repository)
---
 
## System Overview
 
The system operates as a local desktop application built with Python. It combines:
 
- **RetinaFace** for robust face detection across angles and lighting conditions
- **ArcFace** (via InsightFace) for state-of-the-art 512-dimensional face embedding
- **MiniFASNet** (Silent-Face-Anti-Spoofing) for passive texture-based liveness detection
- **MediaPipe Face Mesh** for active challenge–response liveness (blink, head pose, smile, iris tracking)
- **SQLCipher** for AES-256 encrypted storage of biometric templates
- **Cancelable biometric templates** so that a leaked database cannot be used to reconstruct or reuse face embeddings
The threat model covers: print attacks (photo held to camera), replay attacks (video on a screen), and basic 3D mask attacks. Each is addressed by a dedicated countermeasure detailed in the [Security Measures](#security-measures) section.
 
---
 
## Pipeline & System Layers
 
The system processes every access attempt through six sequential stages. A failure at any stage terminates the session immediately.
 
```
┌─────────────────────────────────────────────────────────┐
│                    Access attempt                       │
└──────────────────────┬──────────────────────────────────┘
                       │
               ┌───────▼────────┐
               │  Face detection │  ← RetinaFace
               └───────┬────────┘
                       │  face found
               ┌───────▼────────────┐
               │  Passive liveness  │  ← MiniFASNet (texture)
               └───────┬────────────┘
                       │  real texture confirmed
               ┌───────▼────────────┐
               │  Active liveness   │  ← MediaPipe challenge–response
               └───────┬────────────┘
                       │  challenge passed
               ┌───────▼────────────┐
               │  Embedding extract │  ← ArcFace (InsightFace)
               └───────┬────────────┘
                       │  512-d vector
               ┌───────▼────────────────────┐
               │  Identity matching          │  ← Cosine similarity vs DB
               │  (cancelable template DB)   │  ← SQLCipher (AES-256)
               └───────┬────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   Decision + audit log  │  ← HMAC-chained log
          │   GRANTED  /  DENIED    │
          └─────────────────────────┘
```
 
### Layer descriptions
 
| Layer | Purpose | Defeats |
|---|---|---|
| Face detection | Locate and crop the face region | Frames with no face present |
| Passive liveness | Texture and frequency analysis of the face crop | Print attacks, screen replay |
| Active liveness | Randomised behavioral challenges with time window | Static image attacks, pre-recorded video |
| Embedding extraction | Convert face crop to a 512-d numerical vector | — (representation stage) |
| Identity matching | Compare vector to enrolled templates via cosine similarity | — (decision stage) |
| Audit logging | Tamper-evident record of every access attempt | Log tampering, deniability |
 
---
 
## Project Structure
 
```
secure-face-recognition/
│
├── main.py                        # Entry point — launches the desktop application
├── scaffold.py                    # One-time project scaffolding script
├── requirements.txt               # All pinned dependencies
├── .env.example                   # Template for environment variables (no real secrets)
├── .env                           # Real secrets — gitignored, never committed
├── .gitignore                     # Excludes data/, *.db, *.log, .env, model weights
├── README.md                      # This file
│
├── core/                          # AI pipeline — face detection, embedding, matching
│   ├── __init__.py
│   ├── detector.py                # RetinaFace wrapper: detects and crops face regions
│   │                              #   tools: insightface, opencv-python
│   ├── embedder.py                # ArcFace wrapper: produces 512-d face embeddings
│   │                              #   tools: insightface (ArcFace model), numpy
│   ├── matcher.py                 # Cosine similarity comparison, threshold logic,
│   │                              #   FAR/FRR tracking
│   │                              #   tools: numpy, scikit-learn
│   └── enrollment.py             # Captures N frames at enrollment, averages embeddings,
│                                  #   applies cancelable transform, stores in DB
│                                  #   tools: opencv-python, core/embedder, security/cancelable,
│                                  #          security/database
│
├── liveness/                      # Anti-spoofing — passive texture + active challenges
│   ├── __init__.py
│   ├── passive.py                 # MiniFASNet wrapper: classifies face crop as real or spoof
│   │                              #   tools: Silent-Face-Anti-Spoofing (MiniFASNet),
│   │                              #          opencv-python, torch
│   ├── active.py                  # Challenge orchestrator: randomly samples 2–3 checks,
│   │                              #   enforces 2s time window per challenge, generates
│   │                              #   session token to prevent replay of responses
│   │                              #   tools: random, time, hashlib (session token)
│   └── checks/                    # Individual liveness check implementations
│       ├── __init__.py
│       ├── blink.py               # Eye Aspect Ratio (EAR) formula using eyelid landmarks;
│       │                          #   per-user threshold calibrated at enrollment
│       │                          #   tools: mediapipe, numpy, scipy.spatial.distance
│       ├── head_pose.py           # Pitch, yaw, and roll estimation via solvePnP;
│       │                          #   uses relative angle delta, not absolute degrees
│       │                          #   tools: mediapipe, opencv-python (solvePnP), numpy
│       ├── mouth.py               # Mouth Aspect Ratio (MAR) from inner lip landmarks;
│       │                          #   per-user threshold calibrated at enrollment
│       │                          #   tools: mediapipe, numpy
│       ├── smile.py               # Lip corner spread ratio (landmarks 61 & 291)
│       │                          #   relative to face width
│       │                          #   tools: mediapipe, numpy
│       └── iris.py                # Iris centroid tracking (landmarks 468–477) vs a
│                                  #   moving on-screen target; hardest challenge to spoof
│                                  #   tools: mediapipe, numpy
│
├── security/                      # Privacy & cryptography — storage, keys, audit
│   ├── __init__.py
│   ├── database.py                # SQLCipher encrypted database: CRUD operations for
│   │                              #   student records and transformed embeddings
│   │                              #   tools: sqlcipher3, python-dotenv
│   ├── cancelable.py              # Cancelable biometric templates: applies a secret
│   │                              #   random projection matrix before storage;
│   │                              #   matrix rotation invalidates leaked templates
│   │                              #   tools: numpy
│   ├── key_manager.py             # Reads and writes the DB encryption key via the OS
│   │                              #   credential store — key never touches the filesystem
│   │                              #   tools: keyring
│   └── audit_log.py              # Appends access events to an HMAC-chained log;
│                                  #   each entry includes HMAC of the previous entry
│                                  #   making tampering detectable
│                                  #   tools: hashlib (SHA-256), hmac, json, python-dotenv
│
├── ui/                            # Frontend — CustomTkinter desktop GUI
│   ├── __init__.py
│   ├── app.py                     # Root window and screen router; manages transitions
│   │                              #   between enrollment, liveness, recognition, admin views
│   │                              #   tools: customtkinter
│   ├── camera_widget.py           # Reusable threaded bridge: reads OpenCV frames in a
│   │                              #   background thread and pushes them to a CTk label;
│   │                              #   solves the GUI-freeze problem for all camera screens
│   │                              #   tools: opencv-python, customtkinter, threading
│   ├── enrollment_screen.py       # Live camera feed with face detection overlay;
│   │                              #   capture button saves embedding to encrypted DB
│   │                              #   tools: customtkinter, core/detector, core/enrollment
│   ├── liveness_screen.py         # Displays randomised challenge instructions in large text;
│   │                              #   shows MediaPipe landmark overlay and countdown timer bar
│   │                              #   tools: customtkinter, mediapipe, liveness/active
│   ├── recognition_screen.py      # Live feed with bounding box and identity label;
│   │                              #   shows confidence score and GRANTED/DENIED state
│   │                              #   tools: customtkinter, core/detector, core/embedder,
│   │                              #          core/matcher
│   ├── audit_screen.py            # Scrollable table of access log entries showing
│   │                              #   timestamp, result, and hashed student ID (no PII);
│   │                              #   includes CSV export button
│   │                              #   tools: customtkinter, security/audit_log
│   ├── settings_screen.py         # Similarity threshold slider with live FAR/FRR readout;
│   │                              #   allows examiner to explore the security tradeoff
│   │                              #   tools: customtkinter, core/matcher, evaluation/metrics
│   └── eval_screen.py            # Batch test runner: feeds a test set through the pipeline
│                                  #   and renders ROC curve and EER value inline
│                                  #   tools: customtkinter, evaluation/metrics, matplotlib
│
├── evaluation/                    # Offline metrics — run against test datasets, not live
│   ├── __init__.py
│   ├── metrics.py                 # Computes EER, FAR, FRR at all thresholds; generates
│   │                              #   ROC curve data for report and eval_screen
│   │                              #   tools: scikit-learn, numpy, matplotlib
│   ├── benchmark.py               # Swaps recognition backend (ArcFace / FaceNet / VGG-Face)
│   │                              #   via DeepFace to justify model selection in the report
│   │                              #   tools: deepface, pandas, matplotlib
│   └── attack_sim.py             # Feeds known spoof frames (print photos, screen replays)
│                                  #   through the passive liveness stage to measure detection rate
│                                  #   tools: opencv-python, liveness/passive
│
├── data/                          # Local data — entirely gitignored
│   ├── enrollments/               # Raw captured frames per student (used during enrollment)
│   ├── spoof_samples/             # Print and replay attack frames for evaluation
│   ├── embeddings.db              # SQLCipher AES-256 encrypted embedding database
│   └── audit.log                  # HMAC-chained access event log
│
├── models/                        # Model weight files — gitignored, downloaded at setup
│   ├── .gitkeep                   # Ensures the directory is tracked by Git without weights
│   └── README.md                  # Instructions to download MiniFASNet and InsightFace weights
│
├── tests/                         # Unit tests for security-critical components
│   ├── __init__.py
│   ├── test_matcher.py            # Threshold logic, cosine similarity edge cases,
│   │                              #   impostor vs genuine score distributions
│   ├── test_cancelable.py         # Projection matrix apply/rotate correctness;
│   │                              #   verifies that rotated templates no longer match originals
│   ├── test_audit_log.py          # HMAC chain tamper detection: modifies an entry and
│   │                              #   verifies that chain verification raises an error
│   └── test_liveness.py           # EAR and MAR calculations on known landmark fixtures
│
└── docs/
    ├── threat_model.md            # Per-attack analysis: what each countermeasure defeats
    │                              #   and what it does not; FAR/FRR tradeoff discussion
    ├── architecture.md            # Pipeline diagram, data flow, design decisions and
    │                              #   justifications for tool choices
    └── evaluation_results.md      # ROC curves, EER values, model comparison table,
                                   #   passive liveness detection rates on spoof dataset
```
 
---
 
## Tools, Libraries, Frameworks & Models
 
### Face detection & recognition
 
| Tool | Version | Role |
|---|---|---|
| `insightface` | ≥0.7 | Bundles RetinaFace (detection) and ArcFace (recognition) in one package |
| `onnxruntime` | ≥1.16 | Runtime backend for InsightFace ONNX models |
| `opencv-python` | ≥4.8 | Camera access, image preprocessing, solvePnP head pose, drawing overlays |
| `deepface` | ≥0.0.79 | Benchmark tool — swaps backends (ArcFace, FaceNet, VGG-Face) for model comparison |
 
### Liveness detection
 
| Tool | Version | Role |
|---|---|---|
| `mediapipe` | ≥0.10 | 468-point Face Mesh for EAR, MAR, head pose, smile, and iris landmark extraction |
| `Silent-Face-Anti-Spoofing` | GitHub | MiniFASNet passive liveness classifier (pre-trained on print/replay attacks) |
| `torch` | ≥2.0 | Runtime for MiniFASNet model inference |
| `scipy` | ≥1.11 | Euclidean distance calculations for EAR/MAR landmark pairs |
 
### Security & cryptography
 
| Tool | Version | Role |
|---|---|---|
| `sqlcipher3` | ≥0.5 | AES-256 encrypted SQLite via SQLCipher — transparent, same API as sqlite3 |
| `cryptography` | ≥41 | Fernet (AES-128-CBC + HMAC) for any supplementary encryption operations |
| `keyring` | ≥24 | OS credential store integration for DB key management (never stored on disk) |
| `python-dotenv` | ≥1.0 | Loads environment variables from `.env` at runtime |
| `hmac` + `hashlib` | stdlib | SHA-256 HMAC chain for tamper-evident audit logging |
 
### Data & computation
 
| Tool | Version | Role |
|---|---|---|
| `numpy` | ≥1.24 | Embedding vectors, random projection matrix for cancelable templates, arithmetic |
| `scikit-learn` | ≥1.3 | Cosine similarity, ROC curve, EER computation, confusion matrix |
| `pandas` | ≥2.0 | Benchmark results tabulation, CSV export of audit log |
| `matplotlib` | ≥3.7 | ROC curve rendering in eval_screen and evaluation reports |
 
### Frontend
 
| Tool | Version | Role |
|---|---|---|
| `customtkinter` | ≥5.2 | Modern flat-design Python desktop GUI framework |
| `Pillow` | ≥10 | Converts OpenCV BGR frames to format CustomTkinter can display |
| `threading` | stdlib | Background camera capture thread — prevents GUI freeze |
 
### Models
 
| Model | Source | Used in |
|---|---|---|
| RetinaFace (ResNet-50) | InsightFace model zoo | `core/detector.py` |
| ArcFace (R100, MS1MV3) | InsightFace model zoo | `core/embedder.py` |
| MiniFASNet v2 | Silent-Face-Anti-Spoofing repo | `liveness/passive.py` |
| MediaPipe Face Mesh | Google MediaPipe | All `liveness/checks/` modules |
 
### Development & testing
 
| Tool | Version | Role |
|---|---|---|
| `pytest` | ≥7 | Test runner for all unit tests |
| `python-dotenv` | ≥1.0 | Loads `.env` in test context |
 
---
 
## Security Measures
 
### 1. Cancelable biometric templates
 
**Problem addressed:** If the embedding database is leaked, raw ArcFace embeddings can potentially be used to reconstruct a face image (embedding inversion attacks) or cross-match the individual across other systems using the same model.
 
**Implementation:** Before any embedding is written to the database, it is multiplied by a secret random projection matrix `P` (shape 512×512, drawn from a Gaussian distribution and stored as the system secret). The database stores `P · e` rather than `e`. Matching is performed entirely in the transformed space — the raw embedding never touches the database.
 
**Revocation:** If the database is compromised, the projection matrix is rotated (a new `P` is generated) and all students re-enroll. The leaked templates are now mathematically useless against the new system.
 
**Location:** `security/cancelable.py`
 
---
 
### 2. Passive liveness detection (texture analysis)
 
**Problem addressed:** A printed photograph or a face displayed on a screen has different micro-texture and frequency characteristics than live skin. This defeats simple print and basic replay attacks.
 
**Implementation:** MiniFASNet v2, a lightweight convolutional network pre-trained on the NUAA, CASIA-FASD, and Replay-Attack datasets, classifies each face crop as real or spoof before embedding extraction is attempted. The model analyzes specular highlights, moiré patterns, and edge artifacts that are invisible to the human eye but present in spoofed media.
 
**Limitation:** A high-quality silicone mask with sufficient texture depth may pass this check. It is not the only liveness layer.
 
**Location:** `liveness/passive.py`
 
---
 
### 3. Active liveness detection (challenge–response)
 
**Problem addressed:** A static image, a pre-recorded video, or a basic mask cannot respond dynamically to unpredictable real-time instructions.
 
**Implementation:** The challenge orchestrator randomly samples 2–3 actions from the following pool each session:
 
| Challenge | Detection method | Landmark source |
|---|---|---|
| Eye blink | EAR drops ≥30% below per-user baseline | MediaPipe eyelid landmarks |
| Head yaw (left/right) | Relative yaw angle change via solvePnP | MediaPipe + OpenCV |
| Head pitch (up/down) | Relative pitch angle change via solvePnP | MediaPipe + OpenCV |
| Mouth open | MAR exceeds per-user threshold | MediaPipe inner lip landmarks |
| Smile | Lip corner spread ratio increases | MediaPipe landmarks 61 & 291 |
| Iris tracking | Iris centroid follows moving on-screen dot | MediaPipe iris landmarks 468–477 |
 
Each challenge has a 2-second completion window. A session token (SHA-256 hash of timestamp + random nonce) is generated per session and bound to the challenge sequence, preventing a recorded response from being replayed in a future session.
 
**Per-user calibration:** EAR and MAR thresholds are not global fixed values. They are calibrated from the user's own landmarks during enrollment, making the system robust to anatomical variation (deep-set eyes, monolid eyes, etc.).
 
**Location:** `liveness/active.py`, `liveness/checks/`
 
---
 
### 4. AES-256 encryption at rest
 
**Implementation:** The embedding database (`data/embeddings.db`) is an SQLCipher-encrypted SQLite file. The AES-256 key is never stored on disk — it is read at runtime from the OS credential store via `keyring` (Windows Credential Manager / macOS Keychain / Linux Secret Service). The application receives the key from the environment and opens the database in a single call; the plaintext database file never exists on disk.
 
**Location:** `security/database.py`, `security/key_manager.py`
 
---
 
### 5. Key management
 
**Implementation:** The database encryption key lives exclusively in the OS credential store. It is injected into the application via `keyring.get_password()` at startup. It is never hardcoded in source, never written to `.env` in the committed repository (only the `.env.example` template is committed), and never logged.
 
**Location:** `security/key_manager.py`
 
---
 
### 6. Tamper-evident audit logging
 
**Problem addressed:** An attacker who gains system access might attempt to delete or modify access log entries to erase evidence of failed attempts or unauthorized access.
 
**Implementation:** Every log entry (timestamp, result, hashed student ID, session token) is written with an HMAC (SHA-256, keyed with the DB key) that includes the HMAC of the previous entry. This forms a hash chain: modifying any past entry breaks the chain from that point forward, which is detected at log verification time.
 
Student identifiers in the log are stored as SHA-256 hashes of the student ID — no plaintext names or IDs are ever written to the log.
 
**Location:** `security/audit_log.py`
 
---
 
### 7. Network security (if client–server split is used)
 
If the system is deployed in a client–server configuration, the following additional measures apply:
 
- All communication between camera client and matching server occurs over HTTPS (TLS 1.3)
- Only the 512-d embedding vector is transmitted — the raw face image never crosses the network
- The session token generated by the active liveness orchestrator is included in the request payload to bind the liveness result to the specific embedding being matched
---
 
### Security layer summary
 
| Threat | Countermeasure | Layer |
|---|---|---|
| Photo / printed image | MiniFASNet texture analysis | Passive liveness |
| Screen replay video | MiniFASNet + challenge–response | Passive + active liveness |
| 3D mask | Challenge–response (iris tracking) | Active liveness |
| Pre-recorded response replay | Session token (timestamp + nonce) | Active liveness |
| Stolen embedding database | Cancelable templates + AES-256 | Storage security |
| Embedding inversion attack | Cancelable projection — DB has no raw embeddings | Storage security |
| Log tampering | HMAC chain | Audit integrity |
| Key exposure | OS credential store, never on disk | Key management |
 
---
 
## Setup & Installation
 
### Prerequisites
 
- Python 3.10 or 3.11 (3.12 has known issues with SQLCipher bindings)
- A webcam accessible via OpenCV
- Git
### 1. Clone and enter the repository
 
```bash
git clone https://github.com/your-username/secure-face-recognition.git
cd secure-face-recognition
```
 
### 2. Create and activate a virtual environment
 
```bash
python -m venv .venv
 
# Linux / macOS
source .venv/bin/activate
 
# Windows
.venv\Scripts\activate
```
 
### 3. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Download model weights
 
InsightFace downloads its models automatically on first use. For MiniFASNet:
 
```bash
# Follow the instructions in models/README.md
# or run:
python -c "from silent_face import download_weights; download_weights('models/')"
```
 
### 5. Configure environment
 
```bash
cp .env.example .env
```
 
Open `.env` and set a strong random key for `DB_KEY`. Then store it in the OS credential store:
 
```bash
python -c "
import keyring, os
from dotenv import load_dotenv
load_dotenv()
keyring.set_password('sfrs', 'db_key', os.getenv('DB_KEY'))
print('Key stored.')
"
```
 
 
---
 
## Running the Project
 
```bash
# Launch the desktop application
python main.py
```
 
### Running tests
 
```bash
pytest tests/ -v
```
 
### Running the evaluation benchmark (offline)
 
```bash
python -m evaluation.benchmark
```
 
### Running the attack simulation
 
```bash
python -m evaluation.attack_sim --spoof-dir data/spoof_samples/
```
 
---
 
## What Is Intentionally Excluded from the Repository
 
The following are listed in `.gitignore` and must never be committed:
 
| Excluded item | Reason |
|---|---|
| `data/` | Contains raw face images and the encrypted embedding database — biometric data |
| `*.db` | The SQLCipher database file — even encrypted, biometric databases should not be version-controlled |
| `*.log` | Audit log containing access records |
| `.env` | Contains the database encryption key |
| `models/*.pt`, `models/*.pth`, `models/*.onnx` | Model weight files are large and downloadable from their original sources |
| `__pycache__/` | Compiled Python bytecode |
| `.venv/` | Virtual environment — reproduced from `requirements.txt` |
 
The `models/` directory contains only a `.gitkeep` placeholder and a `README.md` with download instructions. The `data/` directory contains only a `.gitkeep` in each subdirectory.
 
