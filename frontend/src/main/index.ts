import { app, BrowserWindow, ipcMain, shell, dialog } from 'electron'
import { join, resolve, dirname, isAbsolute } from 'path'
import { spawn, ChildProcess } from 'child_process'
import { existsSync, readFileSync, statSync } from 'fs'

let mainWindow: BrowserWindow | null = null
const pythonProcesses: ChildProcess[] = []

// ── Conda locator ─────────────────────────────────────────────────────────

function findCondaExe(): string {
  // 1. Check saved path from setup.bat
  const savedPath = join(resolve(__dirname, '../../../'), '.conda_path')
  if (existsSync(savedPath)) {
    const p = readFileSync(savedPath, 'utf8').trim()
    if (existsSync(p)) return p
  }

  // 2. Common Windows install locations
  const homedir = process.env.USERPROFILE || process.env.HOME || ''
  const candidates = [
    join(homedir, 'miniconda3', 'Scripts', 'conda.exe'),
    join(homedir, 'Miniconda3', 'Scripts', 'conda.exe'),
    join(homedir, 'anaconda3',  'Scripts', 'conda.exe'),
    join(homedir, 'Anaconda3',  'Scripts', 'conda.exe'),
    'C:\\ProgramData\\miniconda3\\Scripts\\conda.exe',
    'C:\\ProgramData\\Miniconda3\\Scripts\\conda.exe',
    'C:\\ProgramData\\anaconda3\\Scripts\\conda.exe',
    'C:\\miniconda3\\Scripts\\conda.exe',
    'C:\\anaconda3\\Scripts\\conda.exe',
  ]
  for (const p of candidates) {
    if (existsSync(p)) return p
  }

  // 3. Fall back to PATH
  return 'conda'
}

// ── Python backend launcher ────────────────────────────────────────────────

// Two independent backends, one per conda env (logic unchanged from the standalone apps):
//   calibration → calib_app on :8765      sync → sync_app on :8766
const PROJECT_ROOT = resolve(__dirname, '../../../')
const BACKENDS = [
  { name: 'calibration', env: 'calib_app', dir: join(PROJECT_ROOT, 'calibration', 'backend') },
  { name: 'sync',        env: 'sync_app',  dir: join(PROJECT_ROOT, 'sync', 'backend') },
]

// Locate <env>/python.exe directly. `conda run` is avoided on purpose: two of them started at once
// race on conda's temp script, and it splits arguments at spaces (any install path containing one).
function findEnvPython(env: string, condaExe: string): string | null {
  const home = process.env.USERPROFILE || process.env.HOME || ''
  const candidates = [
    join(home, '.conda', 'envs', env, 'python.exe'),
    join(dirname(dirname(condaExe)), 'envs', env, 'python.exe'),
    join(home, 'miniconda3', 'envs', env, 'python.exe'),
    join(home, 'anaconda3', 'envs', env, 'python.exe'),
  ]
  return candidates.find(p => existsSync(p)) ?? null
}

function startPythonBackends(): void {
  const condaExe = findCondaExe()

  BACKENDS.forEach((b) => {
    const py = findEnvPython(b.env, condaExe)
    let proc: ChildProcess
    if (py) {
      // Activation equivalent: put the env's DLL folders first on PATH (cv2, onnxruntime, ffmpeg need them).
      const root = dirname(py)
      const path = [root, join(root, 'Library', 'mingw-w64', 'bin'), join(root, 'Library', 'usr', 'bin'),
                    join(root, 'Library', 'bin'), join(root, 'Scripts'), process.env.PATH ?? ''].join(';')
      console.log(`[Main] Starting ${b.name} backend…`, py, 'main.py')
      proc = spawn(py, ['main.py'], { cwd: b.dir, stdio: 'pipe', shell: false, env: { ...process.env, PATH: path, PYTHONUNBUFFERED: '1' } })
    } else {
      // Fallback: conda run (relative script name, staggered so the two runs do not collide)
      console.warn(`[Main] ${b.env} python not found; falling back to "conda run"`)
      proc = spawn(condaExe, ['run', '--no-capture-output', '-n', b.env, 'python', 'main.py'],
        { cwd: b.dir, stdio: 'pipe', shell: false, env: { ...process.env, PYTHONUNBUFFERED: '1' } })
    }
    proc.stdout?.on('data', (d) => console.log(`[${b.name}]`, d.toString().trim()))
    proc.stderr?.on('data', (d) => console.error(`[${b.name}]`, d.toString().trim()))
    proc.on('close', (code) => console.log(`[Main] ${b.name} backend exited with code`, code))
    proc.on('error', (err) => console.error(`[Main] Failed to start ${b.name} backend:`, err))
    pythonProcesses.push(proc)
  })
}

function stopPythonBackends(): void {
  for (const p of pythonProcesses) if (!p.killed) p.kill()
  pythonProcesses.length = 0
}

// ── Window factory ─────────────────────────────────────────────────────────

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width:           1400,
    height:          900,
    minWidth:        1100,
    minHeight:       700,
    backgroundColor: '#080C14',
    frame:           false,
    webPreferences: {
      preload:           join(__dirname, '../preload/index.mjs'),
      contextIsolation:  true,
      nodeIntegration:   false,
      sandbox:           false,
    },
    show: false,
    icon: join(__dirname, '../../resources/icon.png'),
  })

  mainWindow.once('ready-to-show', () => mainWindow?.show())
  mainWindow.on('closed', () => { mainWindow = null })

  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// ── App lifecycle ──────────────────────────────────────────────────────────

app.whenReady().then(() => {
  startPythonBackends()
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  stopPythonBackends()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => stopPythonBackends())

// ── IPC handlers ───────────────────────────────────────────────────────────

ipcMain.handle('open-file-dialog', async (_, options) => {
  const result = await dialog.showOpenDialog(mainWindow!, options)
  return result.filePaths
})

ipcMain.handle('open-folder-dialog', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ['openDirectory'],
  })
  return result.filePaths[0] ?? null
})

// Arguments come from the renderer, so they are validated before reaching the OS shell.
function isHttpUrl(raw: unknown): raw is string {
  if (typeof raw !== 'string') return false
  try {
    const { protocol } = new URL(raw)
    return protocol === 'http:' || protocol === 'https:'
  } catch {
    return false
  }
}

function isExistingAbsolutePath(p: unknown): p is string {
  return typeof p === 'string' && isAbsolute(p) && existsSync(p)
}

ipcMain.handle('open-external', async (_, url: string) => {
  if (!isHttpUrl(url)) throw new Error('Only http(s) URLs can be opened')
  await shell.openExternal(url)
})

ipcMain.handle('open-path', async (_, filePath: string) => {
  // Folders only: opening an arbitrary file would run it with its default program.
  if (!isExistingAbsolutePath(filePath) || !statSync(filePath).isDirectory()) throw new Error('Only existing folders can be opened')
  return shell.openPath(filePath)
})

ipcMain.handle('show-item-in-folder', (_, filePath: string) => {
  if (!isExistingAbsolutePath(filePath)) throw new Error('Path does not exist')
  shell.showItemInFolder(filePath)
})

ipcMain.handle('window-minimize', (event) => {
  BrowserWindow.fromWebContents(event.sender)?.minimize()
})
ipcMain.handle('window-maximize', (event) => {
  const win = BrowserWindow.fromWebContents(event.sender)
  if (!win) return
  win.isMaximized() ? win.unmaximize() : win.maximize()
})
ipcMain.handle('window-close', (event) => {
  BrowserWindow.fromWebContents(event.sender)?.close()
})
ipcMain.handle('window-is-maximized', (event) => {
  return BrowserWindow.fromWebContents(event.sender)?.isMaximized() ?? false
})
ipcMain.handle('backend-status',   () => ({
  running:  pythonProcesses.length > 0 && pythonProcesses.every(p => !p.killed),
  backends: BACKENDS.map((b, i) => ({ name: b.name, pid: pythonProcesses[i]?.pid ?? null })),
}))
