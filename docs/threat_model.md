# Threat model

## Scope

The project is designed to resist common presentation attacks and post-compromise abuse of stored biometric data.

## Addressed threats

### Print attack

A printed face should be rejected by passive liveness before any embedding is extracted.

Countermeasure: [liveness/passive.py](../liveness/passive.py)

### Screen replay attack

A face shown on a phone or monitor should be rejected by passive liveness and then challenged again with active liveness.

Countermeasures: [liveness/passive.py](../liveness/passive.py), [liveness/active.py](../liveness/active.py)

### Basic mask attack

A crude 3D mask may pass one simple check, so the pipeline uses multiple challenge types instead of relying on a single signal.

Countermeasure: [liveness/active.py](../liveness/active.py) plus the checks under [liveness/checks/](../liveness/checks)

### Embedding database leak

Raw embeddings are not stored. The enrolled template is projected through a cancelable transform before storage and matching.

Countermeasures: [security/cancelable.py](../security/cancelable.py), [security/database.py](../security/database.py)

### Log tampering

Audit entries are chained with HMAC so a modified or removed entry breaks chain verification.

Countermeasure: [security/audit_log.py](../security/audit_log.py)

### Key disclosure

The DB key is not committed to the repository or written into the database file.

Countermeasure: [security/key_manager.py](../security/key_manager.py)

### Time-of-Check to Time-of-Use (TOCTOU)

An attacker might attempt to present a photo for recognition, and then swap it for their real face to pass active liveness.

Countermeasure: Identity is verified directly on the frames that pass the active liveness challenges to strongly bind the liveness check to the recognized user. Location: `ui/recognition_screen.py`

## Residual risks

- If the MiniFASNet model is missing, the system will securely fail-closed (deny access) rather than fallback to an unreliable heuristic.
- A user can always be denied service if the camera feed is unavailable or too noisy.
- If SQLCipher is not installed, the repository falls back to sqlite for development, which is not equivalent to the production security target.

## Assumptions

- The camera is local to the desktop app.
- Enrollment is performed by a trusted operator.
- The DB key is provisioned through the OS credential store or the environment during development.

