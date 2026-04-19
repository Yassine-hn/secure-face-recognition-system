# Release Checklist

## Branch hygiene

- [ ] Branch rebased onto latest develop
- [ ] No uncommitted changes
- [ ] CI is green on the release PR

## Test and security validation

- [ ] `python -m unittest -q`
- [ ] `SFRS_MODE=development python main.py --validate-security`
- [ ] `SFRS_MODE=production python main.py --validate-security`

## Runtime prerequisites

- [ ] SQLCipher runtime available (`sqlcipher3` import works)
- [ ] Keyring runtime available (`keyring` import works)
- [ ] DB key set in OS keyring under service `sfrs` and username `db_key`

## Data and secret safety

- [ ] `.env` not committed
- [ ] No biometric samples committed from `data/`
- [ ] No model weights committed from `models/`

## Packaging and release notes

- [ ] Version in `pyproject.toml` reviewed
- [ ] Changelog/release notes drafted
- [ ] Tag name decided (`vX.Y.Z`)

## Post-release smoke checks

- [ ] App launches: `python main.py`
- [ ] Enrollment succeeds for a test identity
- [ ] Recognition logs a GRANTED and a DENIED event
- [ ] Audit export works from the UI
