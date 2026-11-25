# Contributing

Thanks for helping improve ViconMoCap Sync Studio.

## Setup

Follow the install steps in the [README](README.md). The frontend lives in `frontend/`, the two Python services in `calibration/backend/` and `sync/backend/`.

## Before opening a pull request

```bat
cd frontend
npm run typecheck
npm run build

cd ..\sync\backend
python -m pytest

cd ..\..\calibration\backend
python -m pytest
```

- Keep changes focused; one concern per pull request.
- Backend routers stay thin: processing logic belongs in `core/`, request and response models in `models/`.
- Do not change numerical conventions (signed angles, filter cutoffs, coordinate frames, sync sign) without updating [docs/engineering-notes.md](docs/engineering-notes.md) and the tests that lock them.
- Calibration and trial outputs are data, not source: never commit `exports/`, `outputs/`, videos or C3D files.
- UI follows the shared design tokens in `frontend/src/renderer/src/index.css`; use the components under each module's `components/ui`.

## Reporting issues

Use the issue templates. Include the app version, the page, what you expected and what happened, and the backend log lines if any. Do not attach recordings of people.
