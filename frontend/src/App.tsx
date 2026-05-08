import './App.css'

import { useEffect, useMemo, useRef, useState } from 'react'
import DynamicBackground from './DynamicBackground'

type UiStatus =
  | { kind: 'idle' }
  | { kind: 'info'; message: string }
  | { kind: 'error'; message: string }

async function generateComic(params: {
  file: File
  enhance: boolean
  signal?: AbortSignal
}): Promise<Blob> {
  const form = new FormData()
  form.append('file', params.file)
  form.append('preset', 'balanced')
  form.append('enhance', String(params.enhance))

  const res = await fetch('/api/generate', {
    method: 'POST',
    body: form,
    signal: params.signal,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Request failed (${res.status})`)
  }

  const blob = await res.blob()
  if (!blob.type.includes('image')) {
    throw new Error('Server did not return an image')
  }
  return blob
}

async function preprocessForModel(file: File): Promise<File> {
  const canvas = document.createElement('canvas')
  canvas.width = 256
  canvas.height = 256
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    throw new Error('Could not prepare preview canvas')
  }

  if ('createImageBitmap' in window) {
    const imageBitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
    const side = Math.min(imageBitmap.width, imageBitmap.height)
    const sx = Math.floor((imageBitmap.width - side) / 2)
    const sy = Math.floor((imageBitmap.height - side) / 2)
    ctx.drawImage(imageBitmap, sx, sy, side, side, 0, 0, 256, 256)
    imageBitmap.close()
  } else {
    const rawUrl = URL.createObjectURL(file)
    try {
      const img = await new Promise<HTMLImageElement>((resolve, reject) => {
        const el = new Image()
        el.onload = () => resolve(el)
        el.onerror = () => reject(new Error('Failed to decode image'))
        el.src = rawUrl
      })
      const side = Math.min(img.naturalWidth, img.naturalHeight)
      const sx = Math.floor((img.naturalWidth - side) / 2)
      const sy = Math.floor((img.naturalHeight - side) / 2)
      ctx.drawImage(img, sx, sy, side, side, 0, 0, 256, 256)
    } finally {
      URL.revokeObjectURL(rawUrl)
    }
  }

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Failed to encode preview image'))), 'image/png')
  })

  const baseName = file.name.replace(/\.[^.]+$/, '') || 'upload'
  return new File([blob], `${baseName}_256.png`, { type: 'image/png' })
}

function fileToLabel(file: File | null) {
  if (!file) return 'No file selected'
  const kb = file.size / 1024
  const mb = kb / 1024
  const size = mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.round(kb)} KB`
  return `${file.name} • ${size}`
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [enhance, setEnhance] = useState(true)
  const [isDragging, setIsDragging] = useState(false)
  const [isPreparing, setIsPreparing] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [status, setStatus] = useState<UiStatus>({ kind: 'idle' })
  const [outputBlob, setOutputBlob] = useState<Blob | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const inputUrl = useMemo(() => {
    if (!modelFile) return null
    return URL.createObjectURL(modelFile)
  }, [modelFile])

  const outputUrl = useMemo(() => {
    if (!outputBlob) return null
    return URL.createObjectURL(outputBlob)
  }, [outputBlob])

  useEffect(() => {
    return () => {
      if (inputUrl) URL.revokeObjectURL(inputUrl)
    }
  }, [inputUrl])

  useEffect(() => {
    return () => {
      if (outputUrl) URL.revokeObjectURL(outputUrl)
    }
  }, [outputUrl])

  async function setNewFile(next: File | null) {
    setFile(next)
    setModelFile(null)
    setOutputBlob(null)
    if (next) {
      setIsPreparing(true)
      setStatus({ kind: 'info', message: 'Preparing 256x256 model input…' })
      try {
        const prepared = await preprocessForModel(next)
        setModelFile(prepared)
        setStatus({ kind: 'info', message: 'Generating from uploaded photo…' })
      } catch {
        setStatus({ kind: 'error', message: 'Could not prepare image for model input.' })
      } finally {
        setIsPreparing(false)
      }
    } else {
      setStatus({ kind: 'idle' })
    }
  }

  async function onGenerate() {
    if (!modelFile) {
      setStatus({ kind: 'error', message: 'Upload a face photo first.' })
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setIsGenerating(true)
    setStatus({ kind: 'info', message: 'Generating… (runs locally on your machine)' })

    try {
      const blob = await generateComic({ file: modelFile, enhance, signal: controller.signal })
      setOutputBlob(blob)
      setStatus({ kind: 'info', message: 'Done. Download or copy below.' })
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        setStatus({ kind: 'info', message: 'Generation cancelled.' })
      } else {
        const message = err instanceof Error ? err.message : 'Generation failed'
        setStatus({ kind: 'error', message })
      }
    } finally {
      setIsGenerating(false)
    }
  }

  function onCancel() {
    abortRef.current?.abort()
  }

  function onDownload() {
    if (!outputUrl) return
    const a = document.createElement('a')
    a.href = outputUrl
    a.download = `face2comic_${enhance ? 'balanced' : 'raw'}.png`
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  async function onCopyToClipboard() {
    if (!outputBlob) return
    try {
      if (typeof ClipboardItem === 'undefined' || typeof navigator.clipboard?.write !== 'function') {
        throw new Error('Clipboard image copy not supported')
      }
      const item = new ClipboardItem({ 'image/png': outputBlob })
      await navigator.clipboard.write([item])
      setStatus({ kind: 'info', message: 'Copied image to clipboard.' })
    } catch {
      setStatus({ kind: 'error', message: 'Copy failed in this browser. Use Download instead.' })
    }
  }

  function openFilePicker() {
    fileInputRef.current?.click()
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setIsDragging(false)
    const dropped = e.dataTransfer.files?.[0] ?? null
    if (dropped) void setNewFile(dropped)
  }

  useEffect(() => {
    if (!modelFile) return
    void onGenerate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelFile, enhance])

  const canGenerate = !!modelFile && !isGenerating && !isPreparing
  const canCancel = isGenerating

  return (
    <div className="app">
      <DynamicBackground />
      <header className="topbar">
        <div className="brand" aria-label="Face2Comic">
          <div className="mark" aria-hidden="true"></div>
          <div className="brandText">
            <div className="brandName">Face2Comic</div>
            <div className="brandTag">Real photo → comic portrait</div>
          </div>
        </div>
        <div className="topActions">
          <a className="topLink" href="/api/health" target="_blank" rel="noreferrer">
            API
          </a>
        </div>
      </header>

      <main className="container">
        <section className="hero">
          <h1 className="title">Turn your face into a comic</h1>
          <p className="subtitle">
            Upload a face photo and get a generated pix2pix comic portrait instantly.
          </p>
        </section>

        {status.kind !== 'idle' && (
          <div className={`banner ${status.kind}`} role={status.kind === 'error' ? 'alert' : 'status'}>
            <div className="bannerDot" aria-hidden="true"></div>
            <div className="bannerText">{status.message}</div>
          </div>
        )}

        <section className="grid" aria-label="Generator">
          <div className="card">
            <div className="cardHeader">
              <h2>1) Upload</h2>
              <div className="muted">JPG/PNG • shows exact 256x256 model input</div>
            </div>

            <div
              className={`dropzone ${isDragging ? 'dragging' : ''}`}
              role="button"
              tabIndex={0}
              onClick={openFilePicker}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && openFilePicker()}
              onDragEnter={(e) => {
                e.preventDefault()
                setIsDragging(true)
              }}
              onDragOver={(e) => {
                e.preventDefault()
                setIsDragging(true)
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                void onDrop(e)
              }}
              aria-label="Upload image dropzone"
            >
              {inputUrl ? (
                <img className="preview fixedPreview" src={inputUrl} alt="Input preview" />
              ) : (
                <div className="dropzoneEmpty">
                  <div className="dropIcon" aria-hidden="true"></div>
                  <div className="dropTitle">Drop an image here</div>
                  <div className="dropSubtitle">or click to browse</div>
                </div>
              )}
            </div>

            <div className="row">
              <input
                ref={fileInputRef}
                className="fileInput"
                type="file"
                accept="image/*"
                onChange={(e) => {
                  void setNewFile(e.target.files?.[0] ?? null)
                }}
              />
              <div className="fileMeta" aria-label="Selected file">
                {fileToLabel(file)}
              </div>
              {file && (
                <button
                  className="btn ghost"
                  type="button"
                  onClick={() => {
                    void setNewFile(null)
                  }}
                  disabled={isGenerating || isPreparing}
                >
                  Clear
                </button>
              )}
            </div>

            <div className="cardDivider" />

            <div className="controls">
              <label className="toggleField" htmlFor="enhance-toggle">
                <input
                  id="enhance-toggle"
                  className="checkbox"
                  type="checkbox"
                  checked={enhance}
                  onChange={(e) => setEnhance(e.target.checked)}
                  disabled={isGenerating || isPreparing}
                />
                <div className="fieldLabel">Enhance input + output (balanced)</div>
              </label>

              <div className="actions">
                <button className="btn primary" type="button" onClick={onGenerate} disabled={!canGenerate}>
                  {isPreparing ? 'Preparing…' : isGenerating ? 'Generating…' : 'Re-generate'}
                </button>
                <button className="btn" type="button" onClick={onCancel} disabled={!canCancel}>
                  Cancel
                </button>
              </div>
            </div>

            {isGenerating && (
              <div className="progress" aria-label="Progress">
                <div className="progressBar" />
              </div>
            )}
          </div>

          <div className="card">
            <div className="cardHeader">
              <h2>2) Result</h2>
              <div className="muted">Square comic preview</div>
            </div>

            <div className="resultFrame" aria-label="Output preview">
              {outputUrl ? (
                <img className="preview fixedPreview" src={outputUrl} alt="Generated comic" />
              ) : (
                <div className="resultEmpty">
                  <div className="resultTitle">Your comic will appear here</div>
                  <div className="resultSubtitle">Upload a photo to auto-generate</div>
                </div>
              )}
            </div>

            <div className="row">
              <button className="btn primary" type="button" onClick={onDownload} disabled={!outputUrl}>
                Download PNG
              </button>
              <button className="btn" type="button" onClick={onCopyToClipboard} disabled={!outputBlob}>
                Copy
              </button>
              <div className="spacer" />
              <div className="muted small">{enhance ? 'Balanced enhancement enabled' : 'Enhancement disabled'}</div>
            </div>
          </div>
        </section>

        <footer className="footer">
          <div className="muted small">
            Runs locally. No uploads leave your machine unless you deploy this server.
          </div>
        </footer>
      </main>
    </div>
  )
}

export default App
