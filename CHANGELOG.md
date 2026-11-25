# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed

- Electron 29 to 45, electron-builder 24 to 26, electron-vite 2 to 5, Vite 5 to 7. Production dependencies report no known vulnerabilities in `npm audit`. The remaining findings are in packaging tools used at build time only.
- `setup.bat` stops with an error when a conda environment cannot be created or updated.
- README rewritten: overview, pipeline diagram, engineering highlights, requirements, corrected development commands.
- CI runs the backend tests and the Electron build on Windows as well, and uses current action versions.
- `docs/engineering-notes.md` documents why the installer is not supported yet.

### Included in the first version of this repository

- Intrinsic and extrinsic calibration, C3D conversion, synchronization, skeleton overlay, REBA channels, 2D pose and dataset export.
- Local API protections: browser requests only from the app's own origins and loopback host names (extra origins with `VICONSYNC_ALLOWED_ORIGINS`), path containment for file serving and deletion, validated shell requests in the Electron main process.
- Dependency-free tests for both backends and a GitHub Actions workflow that runs them, the frontend typecheck and the builds.
- `SECURITY.md`, pull request template, `setup.bat` with `npm ci` and an FFmpeg check.
- Optional raw Vicon folder lookup, enabled with `VICONSYNC_RAW_VICON_DIR`.
