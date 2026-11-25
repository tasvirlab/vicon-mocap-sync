import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('electron', {
  openFileDialog:   (options: Electron.OpenDialogOptions) =>
    ipcRenderer.invoke('open-file-dialog', options),
  openFolderDialog: () =>
    ipcRenderer.invoke('open-folder-dialog'),
  openExternal:     (url: string) =>
    ipcRenderer.invoke('open-external', url),
  openPath:         (path: string) =>
    ipcRenderer.invoke('open-path', path),
  showItemInFolder: (path: string) =>
    ipcRenderer.invoke('show-item-in-folder', path),
  windowMinimize:    () => ipcRenderer.invoke('window-minimize'),
  windowMaximize:    () => ipcRenderer.invoke('window-maximize'),
  windowClose:       () => ipcRenderer.invoke('window-close'),
  windowIsMaximized: () => ipcRenderer.invoke('window-is-maximized'),
  backendStatus:     () => ipcRenderer.invoke('backend-status'),
})

// Expose platform info
contextBridge.exposeInMainWorld('platform', {
  os: process.platform,
})
