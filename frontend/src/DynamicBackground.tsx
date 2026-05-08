import { useEffect, useRef } from 'react'

type Particle = {
  x: number
  y: number
  vx: number
  vy: number
  r: number
}

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n))
}

function pickCount(width: number, height: number) {
  const area = width * height
  return clamp(Math.round(area / 22000), 26, 90)
}

function parseCssColor(varName: string, fallback: string) {
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(varName)
    .trim()
  return v || fallback
}

export default function DynamicBackground() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (reducedMotion?.matches) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const canvasEl: HTMLCanvasElement = canvas
    const ctx2d: CanvasRenderingContext2D = ctx

    let raf = 0
    let particles: Particle[] = []

    const dotColor = () => parseCssColor('--accent', '#aa3bff')
    const lineColor = () => parseCssColor('--accent-border', 'rgba(170, 59, 255, 0.55)')

    function resize() {
      const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1))
      const { innerWidth: w, innerHeight: h } = window

      canvasEl.width = Math.max(1, Math.floor(w * dpr))
      canvasEl.height = Math.max(1, Math.floor(h * dpr))
      canvasEl.style.width = `${w}px`
      canvasEl.style.height = `${h}px`

      ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0)

      const count = pickCount(w, h)
      particles = Array.from({ length: count }, () => {
        const speed = 0.22 + Math.random() * 0.55
        const angle = Math.random() * Math.PI * 2
        return {
          x: Math.random() * w,
          y: Math.random() * h,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          r: 1 + Math.random() * 1.6,
        }
      })
    }

    function step() {
      const w = window.innerWidth
      const h = window.innerHeight

      ctx2d.clearRect(0, 0, w, h)

      const maxDist = Math.min(220, Math.max(130, Math.round(Math.min(w, h) * 0.22)))
      const maxDist2 = maxDist * maxDist

      const lc = lineColor()
      const dc = dotColor()

      // move
      for (const p of particles) {
        p.x += p.vx
        p.y += p.vy

        if (p.x < -20) p.x = w + 20
        if (p.x > w + 20) p.x = -20
        if (p.y < -20) p.y = h + 20
        if (p.y > h + 20) p.y = -20
      }

      // lines
      ctx2d.lineWidth = 1
      for (let i = 0; i < particles.length; i++) {
        const a = particles[i]!
        for (let j = i + 1; j < particles.length; j++) {
          const b = particles[j]!
          const dx = a.x - b.x
          const dy = a.y - b.y
          const d2 = dx * dx + dy * dy
          if (d2 > maxDist2) continue

          const t = 1 - d2 / maxDist2
          ctx2d.strokeStyle = lc
          ctx2d.globalAlpha = 0.12 + 0.45 * t
          ctx2d.beginPath()
          ctx2d.moveTo(a.x, a.y)
          ctx2d.lineTo(b.x, b.y)
          ctx2d.stroke()
        }
      }

      // dots
      ctx2d.globalAlpha = 0.9
      ctx2d.fillStyle = dc
      for (const p of particles) {
        ctx2d.beginPath()
        ctx2d.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx2d.fill()
      }

      ctx2d.globalAlpha = 1
      raf = window.requestAnimationFrame(step)
    }

    resize()
    raf = window.requestAnimationFrame(step)

    const onResize = () => resize()
    window.addEventListener('resize', onResize, { passive: true })

    return () => {
      window.removeEventListener('resize', onResize)
      window.cancelAnimationFrame(raf)
    }
  }, [])

  return <canvas ref={canvasRef} className="dynamicBg" aria-hidden="true" />
}
