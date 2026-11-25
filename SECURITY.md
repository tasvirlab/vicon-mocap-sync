# Security policy

ViconMoCap Sync Studio is a local desktop application. It has no accounts, no cloud service and no network features beyond downloading the 2D pose models on first use.

## How the local services are protected

- The two FastAPI services bind to `127.0.0.1` and answer only requests whose `Host` is a loopback name.
- Browser requests are accepted only from the app's own pages (`localhost` or `127.0.0.1` on ports 5173 and 5174). Other origins can be added with `VICONSYNC_ALLOWED_ORIGINS`.
- File-serving and delete endpoints check that the resolved path stays inside the project or outputs folder.
- The Electron window runs with context isolation and without Node integration; the main process validates the URLs and paths the renderer asks it to open.

The services have no authentication. Do not expose them to a network (`HOST=0.0.0.0` is not supported) and do not run the app on a shared machine you do not trust.

## Reporting a vulnerability

Use **Security → Report a vulnerability** on the GitHub repository page. If that option is not available, open an issue that says a security report is pending, without technical details, and the maintainer will arrange a private channel.

Please include the version or commit, the steps to reproduce and the impact you observed. Do not attach recordings of people or research data.

## Supported versions

Only the latest commit on `main` receives fixes.
