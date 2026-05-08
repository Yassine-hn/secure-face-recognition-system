# Architecture

## Overview

This system is organized as a local desktop application with three layers:

1. Acquisition and pre-processing: camera frames are collected, cropped, and normalized.
2. Security controls: passive liveness, active liveness, cancelable templates, and encrypted storage.
3. Decision and audit: identity matching writes tamper-evident audit entries for every attempt.

The code now mirrors that flow in practice:

- [core/detector.py](../core/detector.py) detects and crops faces.
- [liveness/passive.py](../liveness/passive.py) scores the crop for spoof likelihood.
- [liveness/active.py](../liveness/active.py) creates a challenge session and executes the selected checks.
- [core/embedder.py](../core/embedder.py) turns a face crop into a normalized embedding.
- [security/cancelable.py](../security/cancelable.py) projects embeddings into a cancelable template.
- [security/database.py](../security/database.py) stores transformed templates in encrypted storage when SQLCipher is available.
- [core/matcher.py](../core/matcher.py) compares the transformed query against enrolled templates.
- [security/audit_log.py](../security/audit_log.py) appends every access attempt to an HMAC-chained log.

## Runtime flow

The UI uses the same sequence:

1. Enrollment captures multiple samples, averages embeddings, and stores a transformed template.
2. Recognition captures a burst of frames, checks passive liveness and averages the scores to reduce noise.
3. If passive liveness passes or requires further verification, the enrolled templates are matched against the best recognized frame.
4. Active liveness may be triggered, during which identity is continuously verified on the specific frames passing the challenges.
4. Every grant or denial is written to the audit log.

## UI structure

The desktop application is routed from [ui/app.py](../ui/app.py). Each screen is isolated, but the screens share one injected service bundle:

- detector
- embedder
- matcher
- recognition service
- passive liveness checker
- active liveness orchestrator
- enrollment service
- audit logger

This keeps the UI thin and moves the logic into testable backend modules.

## Design notes

- Passive and active liveness are independent gates, so one failure can stop the pipeline early.
- Enrollment and recognition both use the same cancelable template representation.
- The app is built to work with injected fakes in tests and real InsightFace/OpenCV components in production.
- The code intentionally avoids storing secrets in files; the DB key is resolved from keyring or environment fallback.

