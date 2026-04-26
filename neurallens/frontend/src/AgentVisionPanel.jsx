import { useMemo, useRef, useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts'

// ── Terminal line builder ──────────────────────────────────────────────────────

function buildLines(type, data, ts) {
  const t = new Date(ts || Date.now()).toLocaleTimeString('en-US', { hour12: false })
  const line = (icon, text, color = 'text-gray-300') => ({ t, icon, text, color })

  if (type === 'progress') {
    switch (data.status) {
      case 'scraping':
        return [line('🔍', `Simulating human browsing: ${data.message || ''}`, 'text-cyan-400')]
      case 'gaze_analysis':
        return [line('👁️', 'Predicting gaze patterns…', 'text-orange-400')]
      case 'scoring':
        return [line('🧬', data.message || 'Waiting for TRIBE scoring response…', 'text-blue-300')]
      case 'scoring_wait':
        return [line('⏳', data.message || 'Still waiting on TRIBE API…', 'text-amber-300')]
      case 'baseline':
        return [line('🧠', `Baseline  overall=${data.score?.overall_score?.toFixed(4)}  L=${data.score?.language_roi?.toFixed(3)}  A=${data.score?.attention_roi?.toFixed(3)}  V=${data.score?.visual_roi?.toFixed(3)}`, 'text-violet-300')]
      case 'proposing': {
        const at = data.action_type || '?'
        return [line('⚡', `[${data.iteration_count}/${data.max_iterations}]  ε=${data.epsilon}  (${data.strategy})  →  ${at}`, 'text-yellow-300')]
      }
      case 'approval_needed': {
        const cur = Number(data.current_overall ?? 0).toFixed(4)
        const prop = Number(data.proposed_overall ?? 0).toFixed(4)
        const d = Number(data.score_delta ?? 0)
        return [line(
          '⏸️',
          `PAUSED  awaiting your decision  current=${cur}  proposed=${prop}  Δ=${d >= 0 ? '+' : ''}${d.toFixed(4)}`,
          'text-violet-300',
        )]
      }
      case 'iteration_complete': {
        const ok = data.accepted
        const d = data.reward ?? 0
        const thought = data.agent_thought || {}
        const lines = [
          line(
            ok ? '✅' : '❌',
            `${ok ? 'ACCEPTED' : 'REJECTED'}  ${d > 0 ? '+' : ''}${d.toFixed(4)}  →  overall=${data.score?.overall_score?.toFixed(4)}`,
            ok ? 'text-green-400' : 'text-red-400',
          ),
        ]
        if (thought.original) {
          lines.push(line('  ', `  ← "${thought.original.slice(0, 72)}"`, 'text-gray-500'))
          lines.push(line('  ', `  → "${thought.replacement?.slice(0, 72)}"`, 'text-gray-400'))
        }
        if (thought.reasoning) {
          lines.push(line('  ', `  💭 ${thought.reasoning.slice(0, 90)}`, 'text-gray-600'))
        }
        return lines
      }
      case 'rendering':
        return [line('🎨', 'Rendering optimized page with accepted edits…', 'text-blue-400')]
      default:
        return []
    }
  }

  if (type === 'gaze') {
    const top = data.gaze_regions?.[0]
    return [line('👁️', `Gaze mapped  ${data.gaze_regions?.length} regions  top saliency=${top?.saliency_score?.toFixed(2) ?? '?'}  (${data.gaze_live ? 'DeepGaze IIE' : 'F-pattern stub'})`, 'text-orange-300')]
  }

  if (type === 'complete') {
    return [
      line('🏁', `DONE  ${data.improvement_pct > 0 ? '+' : ''}${data.improvement_pct?.toFixed(2)}%  improvement  (${data.iterations} iterations)`, 'text-green-300'),
      line('💾', `${data.discovered_patterns} patterns in memory`, 'text-violet-400'),
    ]
  }

  if (type === 'error') {
    return [line('💥', `ERROR: ${data.message}`, 'text-red-400')]
  }

  return []
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function Terminal({ lines }) {
  const bottomRef = useRef(null)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines.length])

  return (
    <div className="flex-1 overflow-y-auto font-mono text-[11px] leading-relaxed bg-gray-950 rounded-lg border border-gray-800 p-3 min-h-0">
      {lines.length === 0 ? (
        <div className="text-gray-700 text-center mt-6">Waiting for optimization to start…</div>
      ) : (
        lines.map((l, i) => (
          <div key={i} className="flex gap-2 hover:bg-gray-900/40 px-1 rounded">
            <span className="text-gray-600 select-none flex-shrink-0">{l.t}</span>
            <span className="select-none w-5 flex-shrink-0">{l.icon}</span>
            <span className={l.color}>{l.text}</span>
          </div>
        ))
      )}
      <div ref={bottomRef} />
    </div>
  )
}

function MiniChart({ chartData }) {
  if (!chartData?.length) return (
    <div className="h-28 flex items-center justify-center text-gray-700 text-xs border border-gray-800 rounded-lg">
      Score chart appears after first iteration
    </div>
  )
  return (
    <div className="h-28 flex-shrink-0">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -28 }}>
          <CartesianGrid strokeDasharray="2 2" stroke="#1f2937" />
          <XAxis dataKey="iteration" tick={{ fill: '#4b5563', fontSize: 9 }} />
          <YAxis domain={[0, 1]} tick={{ fill: '#4b5563', fontSize: 9 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', fontSize: 10, borderRadius: 6 }}
            formatter={v => v.toFixed(4)}
          />
          <Line type="monotone" dataKey="overall"  stroke="#8b5cf6" strokeWidth={2} dot={false} name="Overall" />
          <Line type="monotone" dataKey="language" stroke="#06b6d4" strokeWidth={1} dot={false} name="Lang" strokeDasharray="4 2" />
          <Line type="monotone" dataKey="attention" stroke="#f59e0b" strokeWidth={1} dot={false} name="Attn" strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── AgentVisionPanel ───────────────────────────────────────────────────────────

export default function AgentVisionPanel({
  events,
  chartData,
  status,
  currentIter,
  maxIterations,
  baselineScore,
  finalScore,
  url,
}) {
  const [showHeatmap, setShowHeatmap] = useState(true)
  const imgRef = useRef(null)
  const [imgDims, setImgDims] = useState({ w: 1280, h: 800 })

  // Derive all display data from the events array
  const { gazeHeatmap, iterScreenshot, plainScreenshot, gazeRegions, agentThought, memoryCount, terminalLines } = useMemo(() => {
    let gazeHM = null      // saliency heatmap blend from gaze event
    let iterSS = null      // latest annotated screenshot from iterations/baseline
    let plain = null       // fallback URL from complete event
    let gaze = []
    let thought = null
    let memCount = 0
    const lines = []

    for (const ev of events) {
      const { type, data } = ev

      const newLines = buildLines(type, data, ev.ts)
      lines.push(...newLines)

      if (type === 'gaze') {
        gaze = data.gaze_regions || []
        // Gaze event carries the saliency heatmap blend
        if (data.annotated_screenshot_base64) {
          gazeHM = `data:image/png;base64,${data.annotated_screenshot_base64}`
        }
      }

      if (type === 'progress') {
        // Baseline / iteration_complete carry the latest plain/annotated screenshot
        if (data.annotated_screenshot_base64) {
          iterSS = `data:image/png;base64,${data.annotated_screenshot_base64}`
        }
        if (data.agent_thought) thought = data.agent_thought
        if (data.memory_count != null) memCount = data.memory_count
        if (data.gaze_regions?.length) gaze = data.gaze_regions
      }

      if (type === 'complete' && data.before_screenshot) {
        plain = `/job/${data.job_id}/before-screenshot`
      }
    }

    return {
      gazeHeatmap: gazeHM,
      iterScreenshot: iterSS,
      plainScreenshot: plain,
      gazeRegions: gaze,
      agentThought: thought,
      memoryCount: memCount,
      terminalLines: lines,
    }
  }, [events])

  // When heatmap toggle is on and we have a saliency overlay, show it.
  // Otherwise show the latest iteration screenshot or the before-screenshot fallback.
  const imgSrc = (showHeatmap && gazeHeatmap) ? gazeHeatmap : (iterScreenshot || plainScreenshot || null)
  const progress = maxIterations > 0 ? (currentIter / maxIterations) * 100 : 0

  return (
    <div className="flex flex-col h-full min-h-0 bg-gray-950">
      {/* Split body */}
      <div className="flex flex-1 min-h-0 overflow-hidden">

        {/* ── Left: annotated live page view ── */}
        <div className="flex-1 flex flex-col min-h-0 border-r border-gray-800">
          {/* Overlay toggles */}
          <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-800 flex-shrink-0 bg-gray-900/60">
            <span className="text-xs font-semibold text-gray-400">What The Agent Sees</span>
            <div className="ml-auto flex gap-1.5">
              <button
                onClick={() => setShowHeatmap(v => !v)}
                className={`text-[10px] px-2 py-1 rounded font-medium transition cursor-pointer ${
                  showHeatmap ? 'bg-orange-600/80 text-white' : 'bg-gray-800 text-gray-500 hover:text-gray-300'
                }`}
              >
                🔥 Heatmap
              </button>
              {agentThought?.step && (
                <span className={`text-[10px] px-2 py-1 rounded font-semibold ${
                  agentThought.step === 'accepted' ? 'bg-green-800/60 text-green-300' :
                  agentThought.step === 'rejected' ? 'bg-red-800/60 text-red-300' :
                  'bg-violet-800/60 text-violet-300'
                }`}>
                  {agentThought.step === 'accepted' ? '✅ Accepted' :
                   agentThought.step === 'rejected' ? '❌ Rejected' : '⚡ Proposing'}
                </span>
              )}
            </div>
          </div>

          {/* Screenshot */}
          <div className="flex-1 overflow-auto bg-black relative">
            {imgSrc ? (
              <div className="relative inline-block w-full">
                <img
                  ref={imgRef}
                  src={imgSrc}
                  alt="Agent view"
                  className="w-full block transition-opacity duration-300"
                  onLoad={() => {
                    if (imgRef.current) {
                      setImgDims({ w: imgRef.current.naturalWidth, h: imgRef.current.naturalHeight })
                    }
                  }}
                />
                {/* SVG gaze overlay: ranked circles + scan path */}
                {gazeRegions.length > 0 && (
                  <svg
                    className="absolute inset-0 w-full h-full pointer-events-none"
                    viewBox={`0 0 ${imgDims.w} ${imgDims.h}`}
                    preserveAspectRatio="xMidYMid meet"
                  >
                    {gazeRegions.length > 1 && (
                      <polyline
                        points={gazeRegions.map(r => `${r.peak_coords[0]},${r.peak_coords[1]}`).join(' ')}
                        fill="none" stroke="white" strokeWidth="2"
                        strokeDasharray="10,5" opacity="0.6"
                      />
                    )}
                    {gazeRegions.map(r => (
                      <g key={r.rank}>
                        <circle cx={r.peak_coords[0]} cy={r.peak_coords[1]} r="20" fill="rgba(124,58,237,0.8)" stroke="white" strokeWidth="2" />
                        <text x={r.peak_coords[0]} y={r.peak_coords[1]} textAnchor="middle" dominantBaseline="central"
                          fill="white" fontSize="13" fontWeight="700" fontFamily="system-ui">
                          {r.rank}
                        </text>
                      </g>
                    ))}
                  </svg>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full gap-4 text-gray-600 select-none">
                {status === 'running' || status === 'starting' ? (
                  <>
                    <span className="text-3xl animate-pulse">🧠</span>
                    <span className="text-sm">Scraping page — screenshot incoming…</span>
                  </>
                ) : (
                  <>
                    <span className="text-3xl">🔭</span>
                    <span className="text-sm">Start an optimization to see the agent at work</span>
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ── Right: agent thought process ── */}
        <div className="w-[380px] flex flex-col min-h-0 flex-shrink-0">

          {/* Header */}
          <div className="px-3 py-2 border-b border-gray-800 flex-shrink-0 bg-gray-900/60 flex items-center justify-between">
            <span className="text-xs font-semibold text-gray-400">🤖 Agent Thought Process</span>
            {url && <span className="text-[10px] text-gray-600 truncate max-w-[180px]">{url}</span>}
          </div>

          <div className="flex flex-col flex-1 min-h-0 p-3 gap-3 overflow-hidden">

            {/* Terminal log */}
            <Terminal lines={terminalLines} />

            {/* Mini score chart */}
            <div className="flex-shrink-0">
              <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Score History</div>
              <MiniChart chartData={chartData} />
            </div>

            {/* Memory + patterns stats */}
            <div className="flex-shrink-0 flex gap-2">
              <div className="flex-1 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2">
                <div className="text-[10px] text-gray-500 mb-0.5">Experiences</div>
                <div className="text-lg font-bold text-violet-400 font-mono">{memoryCount}</div>
              </div>
              {baselineScore && finalScore && (
                <div className="flex-1 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2">
                  <div className="text-[10px] text-gray-500 mb-0.5">Improvement</div>
                  <div className={`text-lg font-bold font-mono ${
                    finalScore.overall_score >= baselineScore.overall_score ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {finalScore.overall_score >= baselineScore.overall_score ? '+' : ''}
                    {(((finalScore.overall_score - baselineScore.overall_score) / Math.max(baselineScore.overall_score, 1e-6)) * 100).toFixed(1)}%
                  </div>
                </div>
              )}
            </div>

            {/* Latest agent thought card */}
            {agentThought?.reasoning && (
              <div className="flex-shrink-0 bg-gray-900 border border-gray-800 rounded-lg p-3 text-[11px]">
                <div className="text-gray-500 mb-1 font-semibold">Latest Reasoning</div>
                <p className="text-gray-300 leading-relaxed">{agentThought.reasoning.slice(0, 200)}</p>
                {agentThought.expected_roi_impact && (
                  <div className="flex gap-1.5 mt-2 font-mono">
                    {Object.entries(agentThought.expected_roi_impact).map(([k, v]) => (
                      <span key={k} className={`px-1.5 py-0.5 rounded text-[10px] ${
                        v > 0 ? 'bg-green-900/50 text-green-400' : v < 0 ? 'bg-red-900/50 text-red-400' : 'bg-gray-800 text-gray-500'
                      }`}>
                        {k[0].toUpperCase()}: {v > 0 ? '+' : ''}{Number(v).toFixed(2)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Bottom progress bar ── */}
      <div className="flex-shrink-0 border-t border-gray-800 px-4 py-2 flex items-center gap-4 bg-gray-900/80">
        <span className="text-xs text-gray-500 whitespace-nowrap">
          Iteration {currentIter} / {maxIterations}
        </span>
        <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-violet-500 to-cyan-500 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <span className="text-xs text-gray-500 font-mono whitespace-nowrap">
          {Math.round(progress)}%
        </span>
        {baselineScore && (
          <span className="text-xs text-gray-600 whitespace-nowrap">
            Baseline <span className="text-violet-400 font-mono">{Number(baselineScore.overall_score || 0).toFixed(3)}</span>
          </span>
        )}
        {finalScore && (
          <span className="text-xs text-gray-600 whitespace-nowrap">
            Current <span className="text-green-400 font-mono">{Number(finalScore.overall_score || 0).toFixed(3)}</span>
          </span>
        )}
        <span className={`text-xs font-semibold flex items-center gap-1.5 ${
          status === 'running' ? 'text-cyan-400' : status === 'complete' ? 'text-green-400' : 'text-gray-500'
        }`}>
          {status === 'running' && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse inline-block" />}
          {status === 'complete' && <span className="w-1.5 h-1.5 rounded-full bg-green-400 inline-block" />}
          {status === 'running' ? 'Optimizing…' : status === 'complete' ? 'Complete' : status}
        </span>
      </div>
    </div>
  )
}
