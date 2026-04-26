import { useState, useCallback, useEffect, useRef } from 'react'
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core'
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'

// ── Constants ──────────────────────────────────────────────────────────────────

const COMP_STYLES = {
  headline:    { bg: 'from-violet-900/20', border: 'border-violet-700/40', badge: 'bg-violet-800/60 text-violet-200' },
  cta:         { bg: 'from-cyan-900/20',   border: 'border-cyan-700/40',   badge: 'bg-cyan-800/60 text-cyan-200'   },
  body:        { bg: 'from-gray-900/50',   border: 'border-gray-700/40',   badge: 'bg-gray-700/60 text-gray-300'   },
  testimonial: { bg: 'from-amber-900/20',  border: 'border-amber-700/30',  badge: 'bg-amber-800/40 text-amber-200' },
  image:       { bg: 'from-green-900/20',  border: 'border-green-700/30',  badge: 'bg-green-800/40 text-green-200' },
}
const COMP_ICONS   = { headline: '📰', cta: '🎯', body: '📄', testimonial: '⭐', image: '🖼️' }
const COMP_DEFAULTS = {
  headline:    'Your compelling headline here',
  cta:         'Get Started Free',
  body:        'Add your body copy here. Describe the value proposition clearly and concisely.',
  testimonial: '"This product transformed our business." — Happy Customer',
  image:       '[Image placeholder]',
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function heatColor(score) {
  if (score >= 0.7) return 'bg-green-500/15 border-green-500/40'
  if (score >= 0.4) return 'bg-yellow-500/15 border-yellow-500/40'
  return 'bg-red-500/15 border-red-500/40'
}

function barColor(score) {
  if (score >= 0.7) return 'bg-green-500'
  if (score >= 0.4) return 'bg-yellow-500'
  return 'bg-red-500'
}

/** Map gaze regions to component IDs by proportional vertical position. */
function assignGazeRanks(components, gazeRegions, imageHeight = 800) {
  if (!gazeRegions?.length || !components?.length) return {}
  const rankMap = {}
  const usedIdx = new Set()

  for (const region of gazeRegions) {
    const regionYCenter = (region.bbox[1] + region.bbox[3]) / 2
    const regionYFrac = regionYCenter / imageHeight

    let bestIdx = -1, bestDist = Infinity
    for (let i = 0; i < components.length; i++) {
      if (usedIdx.has(i)) continue
      const compYFrac = (i + 0.5) / components.length
      const dist = Math.abs(compYFrac - regionYFrac)
      if (dist < bestDist) { bestDist = dist; bestIdx = i }
    }
    if (bestIdx >= 0) {
      usedIdx.add(bestIdx)
      rankMap[components[bestIdx].id] = region.rank
    }
  }
  return rankMap
}

// ── Gaze scan-path panel ───────────────────────────────────────────────────────

function GazePanel({ screenshot, gazeOverlay, gazeRegions, gazeLive }) {
  const [showHeatmap, setShowHeatmap] = useState(true)
  const imgRef = useRef(null)
  const [imgDims, setImgDims] = useState({ w: 1280, h: 800 })

  const src = showHeatmap && gazeOverlay
    ? `data:image/png;base64,${gazeOverlay}`
    : screenshot

  const handleLoad = () => {
    if (imgRef.current) {
      setImgDims({ w: imgRef.current.naturalWidth, h: imgRef.current.naturalHeight })
    }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-gray-300">👁️ Gaze Analysis</span>
          <span className={`text-[10px] px-2 py-0.5 rounded-full ${gazeLive ? 'bg-cyan-800/60 text-cyan-200' : 'bg-gray-700/60 text-gray-400'}`}>
            {gazeLive ? 'DeepGaze IIE' : 'F-pattern stub'}
          </span>
          <span className="text-[10px] text-gray-500">{gazeRegions.length} salient regions</span>
        </div>
        {/* Heatmap / Original toggle */}
        <div className="flex rounded-lg overflow-hidden border border-gray-700 text-[11px]">
          {[['🔥 Heatmap', true], ['📄 Original', false]].map(([label, val]) => (
            <button
              key={String(val)}
              onClick={() => setShowHeatmap(val)}
              className={`px-3 py-1.5 transition cursor-pointer font-medium ${
                showHeatmap === val ? 'bg-violet-600 text-white' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Screenshot + SVG scan-path overlay */}
      <div className="relative rounded-lg overflow-hidden border border-gray-700/50 bg-black">
        {src ? (
          <>
            <img
              ref={imgRef}
              src={src}
              alt="Gaze heatmap"
              className="w-full block"
              onLoad={handleLoad}
            />
            {/* SVG scan path — coordinates in native image pixel space */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none"
              viewBox={`0 0 ${imgDims.w} ${imgDims.h}`}
              preserveAspectRatio="xMidYMid meet"
            >
              {/* Dotted path connecting peaks in rank order */}
              {gazeRegions.length > 1 && (
                <polyline
                  points={gazeRegions.map(r => `${r.peak_coords[0]},${r.peak_coords[1]}`).join(' ')}
                  fill="none"
                  stroke="white"
                  strokeWidth={Math.max(2, imgDims.w / 640)}
                  strokeDasharray={`${imgDims.w / 160},${imgDims.w / 320}`}
                  opacity="0.65"
                />
              )}
              {/* Numbered circles at each peak */}
              {gazeRegions.map(r => {
                const r_px = Math.max(14, imgDims.w / 90)
                return (
                  <g key={r.rank}>
                    <circle
                      cx={r.peak_coords[0]} cy={r.peak_coords[1]}
                      r={r_px}
                      fill="rgba(124,58,237,0.82)"
                      stroke="white"
                      strokeWidth={Math.max(1.5, imgDims.w / 853)}
                    />
                    <text
                      x={r.peak_coords[0]} y={r.peak_coords[1]}
                      textAnchor="middle"
                      dominantBaseline="central"
                      fill="white"
                      fontSize={Math.max(10, imgDims.w / 128)}
                      fontWeight="700"
                      fontFamily="system-ui, sans-serif"
                    >
                      {r.rank}
                    </text>
                  </g>
                )
              })}
            </svg>
          </>
        ) : (
          <div className="h-40 flex items-center justify-center text-gray-600 text-sm">
            No screenshot available
          </div>
        )}
      </div>

      {/* Region legend */}
      <div className="mt-3 flex flex-wrap gap-2">
        {gazeRegions.map(r => (
          <div key={r.rank} className="flex items-center gap-1.5 text-[11px] bg-gray-800/60 px-2 py-1 rounded-lg">
            <span className="w-4 h-4 rounded-full bg-violet-600 flex items-center justify-center text-[9px] text-white font-bold flex-shrink-0">
              {r.rank}
            </span>
            <span className="text-gray-400">saliency</span>
            <span className="font-mono text-white">{(r.saliency_score * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Sortable block ─────────────────────────────────────────────────────────────

function SortableBlock({ comp, selected, onSelect, neuralView, gazeView, gazeRank, onRemove }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: comp.id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.45 : 1,
  }

  const cs = COMP_STYLES[comp.type] || COMP_STYLES.body
  const isSelected = selected === comp.id

  let borderClass = cs.border
  if (neuralView) borderClass = heatColor(comp.neural_contribution ?? 0.5)
  if (gazeView && gazeRank === 1) borderClass = 'border-orange-400/70 shadow-orange-900/30'

  return (
    <div
      ref={setNodeRef}
      style={style}
      onClick={() => onSelect(comp.id)}
      className={`relative rounded-xl border p-4 cursor-pointer transition-all group bg-gradient-to-br ${cs.bg} ${borderClass} ${
        isSelected
          ? 'ring-2 ring-violet-500 shadow-lg shadow-violet-900/30'
          : 'hover:border-gray-600'
      } ${gazeView && gazeRank === 1 ? 'shadow-md' : ''}`}
    >
      {/* Drag handle */}
      <div
        {...attributes}
        {...listeners}
        onClick={e => e.stopPropagation()}
        className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-700 hover:text-gray-500 cursor-grab active:cursor-grabbing select-none px-1"
      >
        ⠿
      </div>

      <div className="ml-6 pr-6">
        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
          <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${cs.badge}`}>
            {COMP_ICONS[comp.type]} {comp.type.toUpperCase()}
          </span>

          {/* Neural contribution badge */}
          {neuralView && (
            <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${barColor(comp.neural_contribution ?? 0.5)} text-white`}>
              {((comp.neural_contribution ?? 0.5) * 100).toFixed(0)}
            </span>
          )}

          {/* Gaze rank badge */}
          {gazeView && gazeRank != null && (
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full flex items-center gap-1 ${
              gazeRank === 1 ? 'bg-orange-500/80 text-white' :
              gazeRank === 2 ? 'bg-orange-400/60 text-white' :
              'bg-gray-700/70 text-gray-300'
            }`}>
              👁️ #{gazeRank}
              {gazeRank === 1 && <span className="opacity-80">Most Viewed</span>}
            </span>
          )}

          <span className="text-xs text-gray-600 ml-auto">
            {comp.word_count ?? comp.content.split(' ').length}w
          </span>
        </div>

        <p className={`text-sm leading-relaxed line-clamp-3 ${
          comp.type === 'headline' ? 'font-semibold text-white text-base' : 'text-gray-300'
        }`}>
          {comp.content}
        </p>

        {/* Gaze attention info for top region */}
        {gazeView && gazeRank != null && (
          <div className="mt-1.5 text-[10px] text-gray-500">
            {gazeRank === 1
              ? '🎯 Highest predicted eye attention — optimize this first'
              : `Predicted fixation rank #${gazeRank}`}
          </div>
        )}
      </div>

      <button
        onClick={e => { e.stopPropagation(); onRemove(comp.id) }}
        className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity text-gray-600 hover:text-red-400 text-xs px-1.5 py-0.5 rounded"
      >
        ✕
      </button>
    </div>
  )
}

// ── Editor ─────────────────────────────────────────────────────────────────────

export default function Editor() {
  const [url, setUrl] = useState('')
  const [components, setComponents] = useState([])
  const [selected, setSelected] = useState(null)
  const [neuralView, setNeuralView] = useState(false)
  const [status, setStatus] = useState('idle')
  const [editContent, setEditContent] = useState('')
  const [editType, setEditType] = useState('body')
  const [optimizing, setOptimizing] = useState(false)
  const [suggestion, setSuggestion] = useState(null)
  const [totalScore, setTotalScore] = useState(null)
  const [error, setError] = useState(null)
  const [screenshot, setScreenshot] = useState(null)   // base64 original

  // Gaze state
  const [gazeActive, setGazeActive] = useState(false)
  const [gazeRegions, setGazeRegions] = useState([])
  const [gazeOverlay, setGazeOverlay] = useState(null)
  const [gazeLive, setGazeLive] = useState(false)
  const [gazeLoading, setGazeLoading] = useState(false)

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  const selectedComp = components.find(c => c.id === selected) ?? null

  // Gaze rank map: {compId: rank}
  const gazeRankMap = gazeActive && gazeRegions.length
    ? assignGazeRanks(components, gazeRegions)
    : {}

  useEffect(() => {
    if (selectedComp) {
      setEditContent(selectedComp.content)
      setEditType(selectedComp.type)
    }
  }, [selected])

  const handleDragEnd = useCallback((event) => {
    const { active, over } = event
    if (over && active.id !== over.id) {
      setComponents(prev => {
        const oldIdx = prev.findIndex(c => c.id === active.id)
        const newIdx = prev.findIndex(c => c.id === over.id)
        return arrayMove(prev, oldIdx, newIdx)
      })
    }
  }, [])

  // ── API calls ──────────────────────────────────────────────────────────────

  const parsePage = async () => {
    if (!url.trim()) return
    setStatus('parsing')
    setError(null)
    setSuggestion(null)
    setGazeActive(false)
    setGazeRegions([])
    setGazeOverlay(null)
    try {
      const res = await fetch('/parse-page', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setComponents(data.components ?? [])
      setTotalScore(data.page_score ?? null)
      setSelected(null)
      setScreenshot(
        data.screenshot_base64 ? `data:image/png;base64,${data.screenshot_base64}` : null
      )
      setStatus('ready')
    } catch (err) {
      setError(err.message)
      setStatus('idle')
    }
  }

  const runGazeAnalysis = async () => {
    if (!url.trim()) return
    setGazeLoading(true)
    setError(null)
    try {
      const res = await fetch('/gaze-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setGazeRegions(data.salient_regions ?? [])
      setGazeOverlay(data.heatmap_overlay_base64 ?? null)
      setGazeLive(data.gaze_live ?? false)
      setGazeActive(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setGazeLoading(false)
    }
  }

  const scoreLayout = async () => {
    if (!components.length) return
    setStatus('scoring')
    setError(null)
    try {
      const res = await fetch('/score-layout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ components, url }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setTotalScore(data.total_score)
      const scoreMap = Object.fromEntries(
        (data.per_component ?? []).map(p => [p.id, p.neural_contribution])
      )
      setComponents(prev => prev.map(c => ({
        ...c,
        neural_contribution: scoreMap[c.id] ?? c.neural_contribution,
      })))
    } catch (err) {
      setError(err.message)
    } finally {
      setStatus('ready')
    }
  }

  const optimizeBlock = async () => {
    if (!selectedComp) return
    setOptimizing(true)
    setSuggestion(null)
    setError(null)
    try {
      const res = await fetch('/optimize-block', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ block: selectedComp, url, context: components }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setSuggestion(data.edit)
    } catch (err) {
      setError(err.message)
    } finally {
      setOptimizing(false)
    }
  }

  const applySuggestion = () => {
    if (!suggestion || !selected) return
    const newContent = suggestion.replacement || editContent
    setComponents(prev =>
      prev.map(c => c.id === selected
        ? { ...c, content: newContent, word_count: newContent.split(' ').length }
        : c)
    )
    setEditContent(newContent)
    setSuggestion(null)
  }

  const applyManualEdit = () => {
    if (!selected || (editContent === selectedComp?.content && editType === selectedComp?.type)) return
    setComponents(prev =>
      prev.map(c => c.id === selected
        ? { ...c, content: editContent, type: editType, word_count: editContent.split(' ').length }
        : c)
    )
  }

  const addBlock = (type) => {
    const newComp = {
      id: Math.random().toString(36).slice(2, 10),
      type,
      content: COMP_DEFAULTS[type] ?? '',
      word_count: COMP_DEFAULTS[type]?.split(' ').length ?? 1,
      neural_contribution: 0.5,
    }
    setComponents(prev => [...prev, newComp])
    setSelected(newComp.id)
  }

  const removeBlock = (id) => {
    setComponents(prev => prev.filter(c => c.id !== id))
    if (selected === id) setSelected(null)
  }

  const exportHTML = async () => {
    setError(null)
    try {
      const res = await fetch('/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ components, url }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const html = await res.text()
      const blob = new Blob([html], { type: 'text/html' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = 'neurallens_export.html'
      a.click()
      URL.revokeObjectURL(a.href)
    } catch (err) {
      setError(err.message)
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-800 flex-shrink-0 flex-wrap bg-gray-950/50">
        <input
          type="url"
          value={url}
          onChange={e => setUrl(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && parsePage()}
          placeholder="https://example.com"
          className="flex-1 min-w-0 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-violet-500"
        />
        <button
          onClick={parsePage}
          disabled={!url.trim() || status === 'parsing'}
          className="bg-violet-600 hover:bg-violet-500 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold py-2 px-4 rounded-lg text-sm transition cursor-pointer disabled:cursor-not-allowed whitespace-nowrap"
        >
          {status === 'parsing' ? '⟳ Parsing…' : '🔍 Parse Page'}
        </button>
        <button
          onClick={scoreLayout}
          disabled={!components.length || status === 'scoring'}
          className="bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-white font-medium py-2 px-3 rounded-lg text-sm transition cursor-pointer disabled:cursor-not-allowed whitespace-nowrap"
        >
          {status === 'scoring' ? '⟳ Scoring…' : '🧠 Score Layout'}
        </button>
        <button
          onClick={() => setNeuralView(v => !v)}
          className={`py-2 px-3 rounded-lg text-sm font-medium transition cursor-pointer whitespace-nowrap ${
            neuralView ? 'bg-cyan-700 text-white ring-1 ring-cyan-500' : 'bg-gray-800 hover:bg-gray-700 text-white'
          }`}
        >
          🔥 Neural View
        </button>

        {/* Gaze button */}
        <button
          onClick={gazeActive ? () => setGazeActive(false) : runGazeAnalysis}
          disabled={!url.trim() || gazeLoading}
          className={`py-2 px-3 rounded-lg text-sm font-medium transition cursor-pointer disabled:cursor-not-allowed whitespace-nowrap ${
            gazeActive
              ? 'bg-orange-600 text-white ring-1 ring-orange-400'
              : 'bg-gray-800 hover:bg-gray-700 text-white disabled:opacity-40'
          }`}
        >
          {gazeLoading ? '⟳ Analyzing…' : gazeActive ? '👁️ Gaze ON' : '👁️ Gaze View'}
        </button>

        <button
          onClick={exportHTML}
          disabled={!components.length}
          className="bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-white font-medium py-2 px-3 rounded-lg text-sm transition cursor-pointer disabled:cursor-not-allowed whitespace-nowrap"
        >
          💾 Export HTML
        </button>
      </div>

      {error && (
        <div className="mx-4 mt-2 flex-shrink-0 text-xs text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2 flex justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="ml-3 underline opacity-70 hover:opacity-100">dismiss</button>
        </div>
      )}

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Component palette */}
        <div className="w-[120px] border-r border-gray-800 flex flex-col gap-2 p-3 flex-shrink-0 overflow-y-auto">
          <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">Add Block</div>
          {Object.entries(COMP_ICONS).map(([type, icon]) => (
            <button
              key={type}
              onClick={() => addBlock(type)}
              className="flex flex-col items-center gap-1 py-3 px-1 rounded-xl bg-gray-900 hover:bg-gray-800 border border-gray-800 hover:border-gray-600 transition cursor-pointer"
            >
              <span className="text-lg">{icon}</span>
              <span className="text-[10px] text-gray-400 capitalize">{type}</span>
            </button>
          ))}
        </div>

        {/* Canvas */}
        <div className="flex-1 overflow-y-auto p-4">
          {components.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-600 gap-3 select-none">
              <div className="text-4xl">🏗️</div>
              <p className="text-sm leading-relaxed">
                Enter a URL and click <span className="text-gray-500">Parse Page</span><br />
                or add blocks from the left panel
              </p>
            </div>
          ) : (
            <>
              {/* Score header */}
              {totalScore && (
                <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 rounded-lg bg-gray-900 border border-gray-800 text-xs">
                  <span className="text-gray-400">Neural Score:</span>
                  <span className="font-mono font-bold text-violet-400">{(totalScore.overall_score * 100).toFixed(1)}</span>
                  <span className="text-gray-600 hidden sm:inline">|</span>
                  <span className="text-gray-500">Lang <span className="text-cyan-400 font-mono">{(totalScore.language_roi * 100).toFixed(1)}</span></span>
                  <span className="text-gray-500">Attn <span className="text-amber-400 font-mono">{(totalScore.attention_roi * 100).toFixed(1)}</span></span>
                  <span className="text-gray-500">Vis <span className="text-green-400 font-mono">{(totalScore.visual_roi * 100).toFixed(1)}</span></span>
                  {neuralView && <span className="ml-auto text-[10px] text-cyan-500 bg-cyan-900/20 px-2 py-0.5 rounded">Neural heatmap active</span>}
                  {gazeActive && <span className="ml-auto text-[10px] text-orange-400 bg-orange-900/20 px-2 py-0.5 rounded">Gaze analysis active</span>}
                </div>
              )}

              {/* Gaze panel */}
              {gazeActive && gazeRegions.length > 0 && (
                <div className="mb-4">
                  <GazePanel
                    screenshot={screenshot}
                    gazeOverlay={gazeOverlay}
                    gazeRegions={gazeRegions}
                    gazeLive={gazeLive}
                  />
                </div>
              )}

              {/* Block list */}
              <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                <SortableContext items={components.map(c => c.id)} strategy={verticalListSortingStrategy}>
                  <div className="flex flex-col gap-3">
                    {components.map(comp => (
                      <SortableBlock
                        key={comp.id}
                        comp={comp}
                        selected={selected}
                        onSelect={setSelected}
                        neuralView={neuralView}
                        gazeView={gazeActive}
                        gazeRank={gazeRankMap[comp.id] ?? null}
                        onRemove={removeBlock}
                      />
                    ))}
                  </div>
                </SortableContext>
              </DndContext>
            </>
          )}
        </div>

        {/* Edit panel */}
        <div className="w-72 border-l border-gray-800 flex flex-col p-4 gap-4 flex-shrink-0 overflow-y-auto bg-gray-950/30">
          {selectedComp ? (
            <>
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-sm font-semibold text-white">Edit Block</span>
                  {gazeActive && gazeRankMap[selectedComp.id] != null && (
                    <span className="text-[10px] bg-orange-500/70 text-white px-2 py-0.5 rounded-full font-bold">
                      👁️ Gaze #{gazeRankMap[selectedComp.id]}
                    </span>
                  )}
                </div>

                {/* Type selector */}
                <div className="flex flex-wrap gap-1 mb-3">
                  {Object.keys(COMP_ICONS).map(t => (
                    <button
                      key={t}
                      onClick={() => setEditType(t)}
                      className={`text-[10px] px-2 py-1 rounded-full transition cursor-pointer ${
                        editType === t
                          ? (COMP_STYLES[t] || COMP_STYLES.body).badge
                          : 'bg-gray-800 text-gray-500 hover:text-gray-300'
                      }`}
                    >
                      {COMP_ICONS[t]} {t}
                    </button>
                  ))}
                </div>

                <textarea
                  value={editContent}
                  onChange={e => setEditContent(e.target.value)}
                  rows={6}
                  className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-violet-500 resize-none"
                />

                {/* Neural contribution bar */}
                {selectedComp.neural_contribution !== undefined && (
                  <div className="mt-2.5">
                    <div className="flex justify-between text-xs text-gray-500 mb-1">
                      <span>Neural Contribution</span>
                      <span className="font-mono">{(selectedComp.neural_contribution * 100).toFixed(1)}</span>
                    </div>
                    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${barColor(selectedComp.neural_contribution)}`}
                        style={{ width: `${selectedComp.neural_contribution * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Gaze saliency indicator for selected block */}
                {gazeActive && gazeRankMap[selectedComp.id] != null && (() => {
                  const region = gazeRegions.find(r => r.rank === gazeRankMap[selectedComp.id])
                  return region ? (
                    <div className="mt-2.5">
                      <div className="flex justify-between text-xs text-gray-500 mb-1">
                        <span>Eye Attention (Gaze)</span>
                        <span className="font-mono text-orange-400">{(region.saliency_score * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full bg-orange-500 transition-all"
                          style={{ width: `${region.saliency_score * 100}%` }}
                        />
                      </div>
                    </div>
                  ) : null
                })()}

                <div className="flex gap-2 mt-3">
                  <button
                    onClick={applyManualEdit}
                    disabled={editContent === selectedComp.content && editType === selectedComp.type}
                    className="flex-1 bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white text-xs font-medium py-2 rounded-lg transition cursor-pointer disabled:cursor-not-allowed"
                  >
                    Apply Edit
                  </button>
                  <button
                    onClick={optimizeBlock}
                    disabled={optimizing}
                    className="flex-1 bg-violet-600 hover:bg-violet-500 disabled:bg-gray-800 disabled:text-gray-600 text-white text-xs font-semibold py-2 rounded-lg transition cursor-pointer disabled:cursor-not-allowed"
                  >
                    {optimizing ? '⟳ …' : '⚡ Optimize'}
                  </button>
                </div>
              </div>

              {/* AI suggestion card */}
              {suggestion && (
                <div className="bg-gray-900 border border-violet-700/40 rounded-xl p-3 flex flex-col gap-2">
                  <div className="text-xs font-semibold text-violet-300">AI Suggestion</div>
                  {suggestion.original && (
                    <div className="text-xs text-gray-500 line-through leading-relaxed">
                      {suggestion.original.slice(0, 120)}
                    </div>
                  )}
                  <div className="text-xs text-green-300 leading-relaxed">
                    {suggestion.replacement?.slice(0, 240)}
                  </div>
                  {suggestion.reasoning && (
                    <div className="text-[10px] text-gray-500 italic leading-relaxed">
                      {suggestion.reasoning.slice(0, 180)}
                    </div>
                  )}
                  {suggestion.expected_roi_impact && (
                    <div className="flex gap-2 text-[10px] font-mono">
                      {Object.entries(suggestion.expected_roi_impact).map(([k, v]) => (
                        <span key={k} className={`px-1.5 py-0.5 rounded ${v > 0 ? 'bg-green-900/40 text-green-400' : v < 0 ? 'bg-red-900/40 text-red-400' : 'bg-gray-800 text-gray-500'}`}>
                          {k[0].toUpperCase()}: {v > 0 ? '+' : ''}{Number(v).toFixed(2)}
                        </span>
                      ))}
                    </div>
                  )}
                  <div className="flex gap-2 mt-1">
                    <button
                      onClick={applySuggestion}
                      className="flex-1 bg-green-700 hover:bg-green-600 text-white text-xs font-semibold py-1.5 rounded-lg transition cursor-pointer"
                    >
                      ✓ Accept
                    </button>
                    <button
                      onClick={() => setSuggestion(null)}
                      className="flex-1 bg-gray-800 hover:bg-gray-700 text-gray-400 text-xs py-1.5 rounded-lg transition cursor-pointer"
                    >
                      ✕ Dismiss
                    </button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-600 gap-2 select-none">
              <div className="text-3xl">✏️</div>
              <p className="text-xs leading-relaxed">
                Click a block to edit it<br />or optimize it with AI
              </p>
              {gazeActive && gazeRegions.length > 0 && (
                <p className="text-[10px] text-orange-500/70 mt-2 leading-relaxed">
                  👁️ Gaze active — block #1<br />receives the most eye attention
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
