import { useState, useEffect, useRef, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import Editor from './Editor'
import AgentVisionPanel from './AgentVisionPanel'
import BrainPanel from './BrainPanel'

const API = ''  // proxied through Vite dev server

const ACTION_ICONS = {
  rewrite_headline: '📝',
  rewrite_cta: '🎯',
  rewrite_body_paragraph: '📄',
  adjust_meta_description: '🔍',
  change_visual_hierarchy: '🎨',
  adjust_color_contrast: '🌈',
  reorder_sections: '🔀',
  simplify_language: '✨',
  add_social_proof: '⭐',
  strengthen_value_prop: '💪',
}

const STATUS_MESSAGES = {
  scraping: 'Simulating human browsing…',
  scoring: 'Computing neural activation…',
  proposing: 'Agent proposing edit…',
  baseline: 'Baseline scored',
  iteration_complete: '',
}

export default function App() {
  const [activeTab, setActiveTab] = useState('optimize')  // 'optimize' | 'build' | 'patterns'
  const [url, setUrl] = useState('')
  const [maxIter, setMaxIter] = useState(10)
  const [status, setStatus] = useState('idle')   // idle | starting | running | complete | error
  const [feed, setFeed] = useState([])            // action feed items
  const [chartData, setChartData] = useState([])
  const [currentIter, setCurrentIter] = useState(0)
  const [maxIterations, setMaxIterations] = useState(10)
  const [baselineScore, setBaselineScore] = useState(null)
  const [finalScore, setFinalScore] = useState(null)
  const [acceptedEdits, setAcceptedEdits] = useState([])
  const [error, setError] = useState(null)
  const [previewTab, setPreviewTab] = useState('after')  // 'before' | 'after'
  const [jobIdForPreview, setJobIdForPreview] = useState(null)
  const [memoryStats, setMemoryStats] = useState(null)
  const [events, setEvents] = useState([])               // raw SSE events for AgentVisionPanel
  const [agentVisionActive, setAgentVisionActive] = useState(false)
  const [intent, setIntent] = useState('engage')
  const [brainRegions, setBrainRegions] = useState(null)
  const [ethicsFlags, setEthicsFlags] = useState([])
  const [intentReward, setIntentReward] = useState(null)

  const esRef = useRef(null)
  const feedRef = useRef(null)

  // Auto-scroll feed
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight
    }
  }, [feed])

  // Cleanup SSE on unmount
  useEffect(() => () => esRef.current?.close(), [])

  // Refresh memory stats whenever a run completes
  useEffect(() => {
    if (status !== 'complete') return
    fetch('/memory/stats')
      .then(r => r.ok ? r.json() : null)
      .then(data => data && setMemoryStats(data))
      .catch(() => {})
  }, [status])

  // Auto-activate agent vision when optimization starts
  useEffect(() => {
    if (status === 'running') setAgentVisionActive(true)
  }, [status])

  const handleEvent = useCallback((type, data) => {
    // Accumulate all raw events for AgentVisionPanel
    setEvents(prev => [...prev, { type, data }])
    if (type === 'progress') {
      const { status: s } = data

      if (s === 'baseline' && data.score) {
        setBaselineScore(data.score)
        setChartData([buildChartPoint(0, data.score)])
        setFeed(prev => [...prev, { kind: 'baseline', score: data.score }])
        return
      }

      if (s === 'iteration_complete') {
        setCurrentIter(data.iteration_count)
        setChartData(prev => [...prev, buildChartPoint(data.iteration_count, data.score)])
        setFeed(prev => [...prev, {
          kind: 'iteration',
          iteration: data.iteration_count,
          edit: data.edit,
          reward: data.reward,
          score: data.score,
          accepted: data.accepted,
        }])
        if (data.accepted) {
          setAcceptedEdits(prev => [...prev, data.edit])
        }
        return
      }

      // Generic status messages (scraping / scoring / proposing)
      if (['scraping', 'scoring', 'proposing'].includes(s)) {
        setFeed(prev => {
          const last = prev[prev.length - 1]
          if (last?.kind === 'status' && last.status === s) return prev
          return [...prev, { kind: 'status', status: s, message: data.message }]
        })
      }
    }

    if (type === 'brain_regions') {
      setBrainRegions(data.regions)
      setEthicsFlags(data.ethics_flags || [])
      if (data.intent_reward != null) setIntentReward(data.intent_reward)
    }

    if (type === 'complete') {
      setStatus('complete')
      setFinalScore(data.final_score)
      if (data.accepted_edits) setAcceptedEdits(data.accepted_edits)
      if (data.final_brain_regions) setBrainRegions(data.final_brain_regions)
      if (data.ethics_flags) setEthicsFlags(data.ethics_flags)
      setJobIdForPreview(data.job_id)
      esRef.current?.close()
    }

    if (type === 'progress' && data.status === 'rendering') {
      setFeed(prev => [...prev, { kind: 'status', status: 'rendering', message: data.message }])
    }

    if (type === 'error') {
      setStatus('error')
      setError(data.message)
      esRef.current?.close()
    }
  }, [])

  const startOptimization = async () => {
    if (!url.trim()) return
    esRef.current?.close()

    setError(null)
    setStatus('starting')
    setFeed([])
    setChartData([])
    setEvents([])
    setCurrentIter(0)
    setMaxIterations(maxIter)
    setBaselineScore(null)
    setFinalScore(null)
    setAcceptedEdits([])
    setJobIdForPreview(null)
    setPreviewTab('after')
    setBrainRegions(null)
    setEthicsFlags([])
    setIntentReward(null)

    try {
      const res = await fetch(`${API}/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim(), max_iterations: maxIter, intent }),
      })
      if (!res.ok) throw new Error(`Server error: HTTP ${res.status}`)
      const { job_id } = await res.json()

      setStatus('running')
      const es = new EventSource(`${API}/job/${job_id}/stream`)
      esRef.current = es

      es.onmessage = (e) => {
        const { type, data } = JSON.parse(e.data)
        handleEvent(type, data)
      }
      es.onerror = () => {
        setStatus('error')
        setError('Connection to server lost. Is the backend running?')
        es.close()
      }
    } catch (err) {
      setError(err.message)
      setStatus('idle')
    }
  }

  const progress = maxIterations > 0 ? (currentIter / maxIterations) * 100 : 0
  const isRunning = status === 'running' || status === 'starting'

  return (
    <div className="dark min-h-screen bg-gray-950 text-gray-100 font-sans flex flex-col">
      {/* ── Header ── */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-4 flex-shrink-0">
        <div className="flex items-center gap-3 flex-shrink-0">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center font-bold text-sm select-none">
            N
          </div>
          <div>
            <h1 className="text-lg font-bold text-white leading-tight">NeuralLens</h1>
            <p className="text-xs text-gray-500">Neural-optimized web content</p>
          </div>
        </div>

        {/* Tab navigation */}
        <nav className="flex gap-1 ml-6">
          {[
            { id: 'optimize', label: '⚡ Optimize' },
            { id: 'build',    label: '🏗️ Build'    },
            { id: 'patterns', label: '🔬 Patterns' },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition cursor-pointer ${
                activeTab === tab.id
                  ? 'bg-violet-600 text-white'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Agent Vision toggle — only visible on Optimize tab */}
        {activeTab === 'optimize' && (
          <button
            onClick={() => setAgentVisionActive(v => !v)}
            className={`ml-3 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition cursor-pointer border ${
              agentVisionActive
                ? 'bg-cyan-900/40 border-cyan-700/60 text-cyan-300 hover:bg-cyan-900/60'
                : 'border-gray-700 text-gray-400 hover:text-gray-200 hover:bg-gray-800'
            }`}
          >
            <span className={agentVisionActive ? 'animate-pulse' : ''}>👁️</span>
            Agent Vision
          </button>
        )}

        <div className="ml-auto">
          {status === 'running' && (
            <span className="flex items-center gap-2 text-sm text-cyan-400">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              Optimizing…
            </span>
          )}
          {status === 'complete' && (
            <span className="flex items-center gap-2 text-sm text-green-400">
              <span className="w-2 h-2 rounded-full bg-green-400" />
              Complete
            </span>
          )}
          {status === 'error' && (
            <span className="flex items-center gap-2 text-sm text-red-400">
              <span className="w-2 h-2 rounded-full bg-red-400" />
              Error
            </span>
          )}
        </div>
      </header>

      {/* ── Build tab ── */}
      {activeTab === 'build' && (
        <div className="flex-1 min-h-0 overflow-hidden">
          <Editor />
        </div>
      )}

      {/* ── Patterns tab ── */}
      {activeTab === 'patterns' && (
        <div className="flex-1 min-h-0 overflow-y-auto p-6">
          <PatternsTab />
        </div>
      )}

      {/* ── Optimize tab — Agent Vision mode ── */}
      {activeTab === 'optimize' && agentVisionActive && (
        <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
          {/* Compact URL bar at top when vision is active */}
          <div className="flex items-center gap-3 px-4 py-2.5 border-b border-gray-800 flex-shrink-0 bg-gray-950/80">
            <input
              type="url"
              value={url}
              onChange={e => setUrl(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !isRunning && startOptimization()}
              placeholder="https://example.com"
              disabled={isRunning}
              className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 disabled:opacity-50 transition"
            />
            <div className="flex items-center gap-2 flex-shrink-0">
              <span className="text-xs text-gray-500">Iters:</span>
              <input
                type="number"
                value={maxIter}
                onChange={e => setMaxIter(Math.max(1, Math.min(20, parseInt(e.target.value) || 10)))}
                min={1} max={20}
                disabled={isRunning}
                className="w-14 bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-white text-center focus:outline-none focus:border-violet-500 disabled:opacity-50"
              />
            </div>
            <button
              onClick={startOptimization}
              disabled={!url.trim() || isRunning}
              className="bg-violet-600 hover:bg-violet-500 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold py-1.5 px-4 rounded-lg text-sm transition cursor-pointer disabled:cursor-not-allowed flex-shrink-0"
            >
              {status === 'starting' ? 'Starting…' : isRunning ? 'Running…' : '⚡ Optimize'}
            </button>
            {error && (
              <div className="text-xs text-red-400 truncate max-w-xs">{error}</div>
            )}
          </div>
          {/* Full-height Agent Vision Panel */}
          <div className="flex-1 min-h-0">
            <AgentVisionPanel
              events={events}
              chartData={chartData}
              status={status}
              currentIter={currentIter}
              maxIterations={maxIterations}
              baselineScore={baselineScore}
              finalScore={finalScore}
              url={url}
            />
          </div>
        </div>
      )}

      {/* ── Optimize tab — normal split layout ── */}
      {activeTab !== 'build' && activeTab !== 'patterns' && !agentVisionActive && (
      <div className="flex flex-1 min-h-0">
        {/* ── Left panel ── */}
        <div className="w-[40%] border-r border-gray-800 flex flex-col p-5 gap-4 min-h-0">
          {/* URL input */}
          <div className="flex flex-col gap-2.5 flex-shrink-0">
            <label className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
              Target URL
            </label>
            <input
              type="url"
              value={url}
              onChange={e => setUrl(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !isRunning && startOptimization()}
              placeholder="https://example.com"
              disabled={isRunning}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3.5 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 disabled:opacity-50 transition"
            />
            <div className="flex gap-2 items-center">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500 whitespace-nowrap">Iterations:</span>
                <input
                  type="number"
                  value={maxIter}
                  onChange={e => setMaxIter(Math.max(1, Math.min(20, parseInt(e.target.value) || 10)))}
                  min={1}
                  max={20}
                  disabled={isRunning}
                  className="w-14 bg-gray-900 border border-gray-700 rounded px-2 py-2 text-sm text-white text-center focus:outline-none focus:border-violet-500 disabled:opacity-50"
                />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500 whitespace-nowrap">Intent:</span>
                <select
                  value={intent}
                  onChange={e => setIntent(e.target.value)}
                  disabled={isRunning}
                  className="bg-gray-900 border border-gray-700 rounded px-2 py-2 text-xs text-white focus:outline-none focus:border-violet-500 disabled:opacity-50 cursor-pointer"
                >
                  <option value="engage">Engage</option>
                  <option value="trust">Trust</option>
                  <option value="convert">Convert</option>
                  <option value="accessibility">Accessibility</option>
                  <option value="gamification">Gamification</option>
                </select>
              </div>
              <button
                onClick={startOptimization}
                disabled={!url.trim() || isRunning}
                className="flex-1 bg-violet-600 hover:bg-violet-500 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold py-2.5 px-4 rounded-lg text-sm transition cursor-pointer disabled:cursor-not-allowed"
              >
                {status === 'starting' ? 'Starting…' : isRunning ? 'Running…' : '⚡ Optimize'}
              </button>
            </div>

            {error && (
              <div className="text-xs text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2">
                {error}
              </div>
            )}
          </div>

          {/* Progress bar */}
          {(isRunning || status === 'complete') && (
            <div className="flex-shrink-0 flex flex-col gap-1.5">
              <div className="flex justify-between text-xs text-gray-400">
                <span>Iteration {currentIter} / {maxIterations}</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-violet-500 to-cyan-500 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Action feed */}
          <div className="flex flex-col min-h-0 flex-1">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 flex-shrink-0">
              Action Feed
            </h3>
            <div ref={feedRef} className="flex-1 overflow-y-auto flex flex-col gap-2 pr-1">
              {feed.length === 0 && status === 'idle' && (
                <div className="text-xs text-gray-600 italic text-center mt-10 px-4">
                  Enter a URL and click Optimize to begin neural analysis
                </div>
              )}
              {feed.map((item, idx) => (
                <FeedItem key={idx} item={item} />
              ))}
            </div>
          </div>
        </div>

        {/* ── Right panel ── */}
        <div className="flex-1 flex flex-col p-5 gap-5 overflow-y-auto min-h-0">
          {/* Score chart */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex-shrink-0">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-4">
              Neural Activation — Score Over Iterations
            </h3>
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: -15 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis
                    dataKey="iteration"
                    tick={{ fill: '#6b7280', fontSize: 11 }}
                    label={{ value: 'Iteration', position: 'insideBottom', offset: -2, fill: '#4b5563', fontSize: 11 }}
                  />
                  <YAxis domain={[0, 1]} tick={{ fill: '#6b7280', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px', fontSize: '12px' }}
                    labelStyle={{ color: '#9ca3af' }}
                    formatter={(v, name) => [v.toFixed(4), name]}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                  <Line type="monotone" dataKey="overall" stroke="#8b5cf6" strokeWidth={2.5} dot={{ r: 3, fill: '#8b5cf6' }} name="Overall" />
                  <Line type="monotone" dataKey="language" stroke="#06b6d4" strokeWidth={1.5} dot={false} name="Language ROI" strokeDasharray="5 3" />
                  <Line type="monotone" dataKey="attention" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Attention ROI" strokeDasharray="5 3" />
                  <Line type="monotone" dataKey="visual" stroke="#10b981" strokeWidth={1.5} dot={false} name="Visual ROI" strokeDasharray="5 3" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChart />
            )}
          </div>

          {/* Brain activation panel */}
          {(brainRegions || status === 'running' || status === 'complete') && (
            <div className="flex-shrink-0">
              <BrainPanel
                regions={brainRegions}
                ethicsFlags={ethicsFlags}
                intent={intent}
                intentReward={intentReward}
              />
            </div>
          )}

          {/* Memory / Learning panel */}
          {memoryStats && Object.keys(memoryStats).length > 0 && (
            <MemoryPanel stats={memoryStats} />
          )}

          {/* Before / After diff */}
          {acceptedEdits.length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex-shrink-0">
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">
                Accepted Changes — Before / After
              </h3>
              <div className="flex flex-col gap-3 max-h-72 overflow-y-auto pr-1">
                {acceptedEdits.map((edit, idx) => (
                  <DiffCard key={idx} edit={edit} index={idx} />
                ))}
              </div>
            </div>
          )}

          {/* Summary card */}
          {status === 'complete' && baselineScore && finalScore && (
            <SummaryCard
              baseline={baselineScore}
              final={finalScore}
              iterations={maxIterations}
              accepted={acceptedEdits.length}
            />
          )}

          {/* Before / After page preview */}
          {status === 'complete' && jobIdForPreview && (
            <PreviewPanel
              jobId={jobIdForPreview}
              tab={previewTab}
              onTab={setPreviewTab}
            />
          )}

          {/* Landing state */}
          {status === 'idle' && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center max-w-sm">
                <div className="text-5xl mb-5">🧠</div>
                <h2 className="text-lg font-semibold text-gray-300 mb-2">How NeuralLens Works</h2>
                <ol className="text-sm text-gray-500 text-left space-y-2 list-none">
                  {[
                    ['🎬', 'Simulates a human scrolling your page via Playwright'],
                    ['🧬', 'Passes video + text + audio to TRIBE v2 brain encoder'],
                    ['🧠', 'Scores 9 HCP-MMP1 regions: Amygdala, Hippocampus, NAcc + 6 more'],
                    ['🤖', 'Claude agent proposes targeted text edits based on gaze + brain data'],
                    ['🔁', 'Iterates using intent-aware neural reward signal'],
                    ['📈', 'Accepts edits that improve brain engagement within ethical guardrails'],
                  ].map(([icon, text]) => (
                    <li key={text} className="flex items-start gap-2">
                      <span className="mt-0.5">{icon}</span>
                      <span>{text}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          )}
        </div>
      </div>
      )}
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function FeedItem({ item }) {
  if (item.kind === 'baseline') {
    return (
      <div className="px-3 py-2 rounded-lg text-xs bg-violet-900/20 border border-violet-700/30">
        <span className="text-violet-400 font-semibold">Baseline</span>
        <span className="text-gray-400 ml-2">
          overall = <span className="text-white font-mono">{item.score?.overall_score?.toFixed(4)}</span>
        </span>
      </div>
    )
  }

  if (item.kind === 'status') {
    return (
      <div className="px-3 py-1.5 text-xs text-gray-500 flex items-center gap-2">
        <span className="animate-spin inline-block">⟳</span>
        {item.message}
      </div>
    )
  }

  if (item.kind === 'iteration') {
    const icon = ACTION_ICONS[item.edit?.action_type] || '🔧'
    const delta = item.reward ?? 0
    const positive = delta >= 0
    return (
      <div className={`flex flex-col gap-1 px-3 py-2 rounded-lg text-xs border ${
        item.accepted
          ? 'bg-green-900/10 border-green-800/30'
          : 'bg-red-900/10 border-red-800/30'
      }`}>
        <div className="flex items-center justify-between">
          <span className="flex items-center gap-1.5 text-gray-200 font-medium">
            <span>{icon}</span>
            <span className="capitalize">{(item.edit?.action_type || '').replace(/_/g, ' ')}</span>
          </span>
          <span className={`font-mono text-xs ${positive ? 'text-green-400' : 'text-red-400'}`}>
            {positive ? '+' : ''}{delta.toFixed(4)}
          </span>
        </div>
        {item.edit?.target && (
          <div className="text-gray-500 truncate" title={item.edit.target}>{item.edit.target}</div>
        )}
        <span className={`text-xs px-1.5 py-0.5 rounded w-fit font-medium ${
          item.accepted
            ? 'bg-green-800/40 text-green-300'
            : 'bg-red-800/40 text-red-300'
        }`}>
          {item.accepted ? '✓ Accepted' : '✗ Rejected'}
        </span>
      </div>
    )
  }

  return null
}

function DiffCard({ edit, index }) {
  const maxLen = 140
  const truncate = (s) => s && s.length > maxLen ? s.slice(0, maxLen) + '…' : s

  return (
    <div className="text-xs bg-gray-800/50 rounded-lg p-3 border border-gray-700/40">
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <span className="text-gray-500 font-mono">{index + 1}.</span>
        <span className="capitalize text-gray-200 font-semibold">
          {(edit.action_type || '').replace(/_/g, ' ')}
        </span>
        {edit.target && (
          <span className="text-gray-500 truncate max-w-[200px]" title={edit.target}>
            → {edit.target}
          </span>
        )}
        {edit.expected_roi && (
          <span className="ml-auto text-cyan-500 font-mono text-[10px] bg-cyan-900/20 px-1.5 py-0.5 rounded">
            {edit.expected_roi}
          </span>
        )}
      </div>

      {edit.original && (
        <div className="flex flex-col gap-1">
          <div className="line-through text-red-400/80 bg-red-950/30 border border-red-900/30 px-2 py-1.5 rounded break-words leading-relaxed">
            {truncate(edit.original)}
          </div>
          <div className="text-green-300/90 bg-green-950/30 border border-green-900/30 px-2 py-1.5 rounded break-words leading-relaxed">
            {truncate(edit.replacement)}
          </div>
        </div>
      )}

      {edit.reasoning && (
        <div className="text-gray-500 mt-2 italic leading-relaxed">
          {truncate(edit.reasoning)}
        </div>
      )}
    </div>
  )
}

function SummaryCard({ baseline, final, iterations, accepted }) {
  const delta = final.overall_score - baseline.overall_score
  const pct = baseline.overall_score > 0
    ? ((delta / baseline.overall_score) * 100)
    : 0
  const improved = delta >= 0

  return (
    <div className="bg-gradient-to-br from-violet-900/25 to-cyan-900/25 border border-violet-700/30 rounded-xl p-5 flex-shrink-0">
      <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-wide mb-4">
        Optimization Summary
      </h3>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <ScoreStat label="Baseline" value={baseline.overall_score} />
        <ScoreStat label="Final" value={final.overall_score} highlight />
        <div className="text-center">
          <div className={`text-3xl font-bold ${improved ? 'text-green-400' : 'text-red-400'}`}>
            {improved ? '+' : ''}{pct.toFixed(1)}%
          </div>
          <div className="text-xs text-gray-400 mt-1">Improvement</div>
        </div>
      </div>

      <div className="border-t border-gray-700/40 pt-3 grid grid-cols-3 gap-2 text-center">
        {[
          [iterations, 'Iterations'],
          [accepted, 'Accepted'],
          [iterations - accepted, 'Rejected'],
        ].map(([v, label]) => (
          <div key={label}>
            <div className="text-lg font-bold text-white">{v}</div>
            <div className="text-xs text-gray-500">{label}</div>
          </div>
        ))}
      </div>

      <div className="mt-3 pt-3 border-t border-gray-700/40">
        <div className="grid grid-cols-3 gap-2 text-center text-xs">
          {[
            ['Language ROI', final.language_roi],
            ['Attention ROI', final.attention_roi],
            ['Visual ROI', final.visual_roi],
          ].map(([label, val]) => (
            <div key={label} className="bg-gray-800/40 rounded-lg px-2 py-1.5">
              <div className="text-white font-mono font-semibold">{val?.toFixed(3) ?? '—'}</div>
              <div className="text-gray-500 mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function ScoreStat({ label, value, highlight }) {
  return (
    <div className="text-center">
      <div className={`text-3xl font-bold ${highlight ? 'text-violet-300' : 'text-gray-300'}`}>
        {(value * 100).toFixed(1)}
      </div>
      <div className="text-xs text-gray-400 mt-1">{label}</div>
      <div className="text-[10px] text-gray-600">× 100</div>
    </div>
  )
}

function EmptyChart() {
  return (
    <div className="h-[220px] flex flex-col items-center justify-center text-gray-600 gap-2">
      <div className="text-3xl">📊</div>
      <div className="text-sm">Score chart will appear once optimization starts</div>
    </div>
  )
}

function MemoryPanel({ stats }) {
  const rows = Object.entries(stats).sort((a, b) => b[1].avg_reward - a[1].avg_reward)

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex-shrink-0">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
          🧠 Agent Memory — Cross-Session Learning
        </h3>
        <span className="text-[10px] text-gray-600 bg-gray-800 px-2 py-1 rounded">
          {rows.length} action types tracked
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 border-b border-gray-800">
              <th className="text-left py-1.5 pr-3 font-medium">Action Type</th>
              <th className="text-right py-1.5 px-3 font-medium">Avg Reward</th>
              <th className="text-right py-1.5 px-3 font-medium">Success Rate</th>
              <th className="text-right py-1.5 pl-3 font-medium">Tries</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([actionType, s]) => {
              const positive = s.avg_reward > 0
              return (
                <tr
                  key={actionType}
                  className={`border-b border-gray-800/50 transition-colors ${
                    positive ? 'hover:bg-green-950/20' : 'hover:bg-red-950/20'
                  }`}
                >
                  <td className="py-2 pr-3 flex items-center gap-2">
                    <span>{ACTION_ICONS[actionType] || '🔧'}</span>
                    <span className="capitalize text-gray-300">
                      {actionType.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td className={`text-right px-3 font-mono font-semibold ${
                    positive ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {positive ? '+' : ''}{s.avg_reward.toFixed(4)}
                  </td>
                  <td className="text-right px-3">
                    <span className={`inline-flex items-center gap-1 ${
                      s.success_rate >= 0.5 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {(s.success_rate * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="text-right pl-3 text-gray-500 font-mono">{s.count}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] text-gray-600 mt-2">
        GPT-4.1-nano reads these stats before each edit and avoids action types with negative avg reward.
      </p>
    </div>
  )
}

function PreviewPanel({ jobId, tab, onTab }) {
  const [imgError, setImgError] = useState({ before: false, after: false })
  const [imgLoaded, setImgLoaded] = useState({ before: false, after: false })

  const beforeSrc = `/job/${jobId}/before-screenshot`
  const afterSrc  = `/job/${jobId}/after-screenshot`

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex-shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
          Optimized Page Preview
        </h3>
        {/* Tab switcher */}
        <div className="flex rounded-lg overflow-hidden border border-gray-700 text-xs">
          {['before', 'after'].map(t => (
            <button
              key={t}
              onClick={() => onTab(t)}
              className={`px-3 py-1.5 font-medium transition capitalize ${
                tab === t
                  ? t === 'after'
                    ? 'bg-violet-600 text-white'
                    : 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
              }`}
            >
              {t === 'after' ? '✨ After' : '📄 Before'}
            </button>
          ))}
        </div>
      </div>

      {/* Screenshot viewer */}
      <div className="relative rounded-lg overflow-hidden border border-gray-700/50 bg-gray-950" style={{ height: '520px' }}>
        {/* Before */}
        <div className={`absolute inset-0 overflow-y-auto transition-opacity duration-300 ${tab === 'before' ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
          {!imgLoaded.before && !imgError.before && (
            <div className="flex items-center justify-center h-full gap-3 text-gray-500 text-sm">
              <span className="animate-spin">⟳</span> Loading original screenshot…
            </div>
          )}
          {imgError.before && (
            <div className="flex items-center justify-center h-full text-gray-600 text-sm">
              Screenshot not available
            </div>
          )}
          <img
            src={beforeSrc}
            alt="Original page"
            className={`w-full block ${imgLoaded.before ? '' : 'hidden'}`}
            onLoad={() => setImgLoaded(s => ({ ...s, before: true }))}
            onError={() => setImgError(s => ({ ...s, before: true }))}
          />
        </div>

        {/* After */}
        <div className={`absolute inset-0 overflow-y-auto transition-opacity duration-300 ${tab === 'after' ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
          {!imgLoaded.after && !imgError.after && (
            <div className="flex items-center justify-center h-full gap-3 text-gray-500 text-sm">
              <span className="animate-spin">⟳</span> Loading optimized screenshot…
            </div>
          )}
          {imgError.after && (
            <div className="flex items-center justify-center h-full text-gray-600 text-sm">
              Optimized screenshot not available
            </div>
          )}
          <img
            src={afterSrc}
            alt="Optimized page"
            className={`w-full block ${imgLoaded.after ? '' : 'hidden'}`}
            onLoad={() => setImgLoaded(s => ({ ...s, after: true }))}
            onError={() => setImgError(s => ({ ...s, after: true }))}
          />
        </div>

        {/* Badge overlay */}
        <div className="absolute top-2 left-2 pointer-events-none">
          {tab === 'before' ? (
            <span className="text-[10px] font-semibold px-2 py-1 rounded bg-gray-800/90 text-gray-400 border border-gray-700">
              ORIGINAL
            </span>
          ) : (
            <span className="text-[10px] font-semibold px-2 py-1 rounded bg-violet-800/90 text-violet-200 border border-violet-700">
              NEURAL-OPTIMIZED
            </span>
          )}
        </div>
      </div>

      <p className="text-[11px] text-gray-600 mt-2 text-center">
        Scroll inside the preview to see the full page. Accepted edits were injected into a live Playwright render.
      </p>
    </div>
  )
}

// ── Patterns tab ──────────────────────────────────────────────────────────────

const PTYPE_COLORS = {
  LEXICAL:       'bg-violet-800/40 text-violet-200 border-violet-700/40',
  SOCIAL_PROOF:  'bg-cyan-800/40 text-cyan-200 border-cyan-700/40',
  COGNITIVE_LOAD:'bg-amber-800/40 text-amber-200 border-amber-700/40',
  STRUCTURAL:    'bg-green-800/40 text-green-200 border-green-700/40',
}

function PatternsTab() {
  const [patterns, setPatterns] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/patterns')
      .then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then(data => { setPatterns(data); setLoading(false) })
      .catch(err => { setError(String(err)); setLoading(false) })
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 gap-3 text-gray-500">
        <span className="animate-spin text-xl">⟳</span>
        Loading patterns…
      </div>
    )
  }

  if (error) {
    return <div className="text-red-400 text-sm p-4">{error}</div>
  }

  if (!patterns.length) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-center text-gray-600 gap-3">
        <div className="text-4xl">🔬</div>
        <p className="text-sm">No patterns discovered yet.<br />Run a few optimizations to start building the knowledge base.</p>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-bold text-white">Learned Patterns</h2>
          <p className="text-xs text-gray-500 mt-0.5">
            Statistical correlations discovered from {patterns.reduce((s, p) => s + p.sample_count, 0)} optimization experiences
          </p>
        </div>
        <span className="text-sm font-mono bg-gray-900 border border-gray-800 px-3 py-1.5 rounded-lg text-violet-400">
          {patterns.length} patterns
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {patterns.map(p => {
          const typeColor = PTYPE_COLORS[p.pattern_type] || 'bg-gray-800/40 text-gray-300 border-gray-700/40'
          const confPct = Math.round((p.confidence ?? 0) * 100)
          const avgDelta = p.avg_overall_delta ?? 0
          const positive = avgDelta >= 0

          return (
            <div key={p.id} className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex flex-col gap-3">
              {/* Header */}
              <div className="flex items-start justify-between gap-2">
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${typeColor}`}>
                  {p.pattern_type}
                </span>
                <span className={`text-xs font-mono font-bold ${positive ? 'text-green-400' : 'text-red-400'}`}>
                  {positive ? '+' : ''}{avgDelta.toFixed(4)} avg Δ
                </span>
              </div>

              {/* Description */}
              <p className="text-xs text-gray-300 leading-relaxed">{p.pattern_description}</p>

              {/* ROI deltas */}
              <div className="flex gap-2 text-[10px] font-mono">
                {[
                  ['L', p.avg_language_roi_delta,  'text-cyan-400'],
                  ['A', p.avg_attention_roi_delta,  'text-amber-400'],
                  ['V', p.avg_visual_roi_delta,     'text-green-400'],
                ].map(([label, val, color]) => (
                  <span key={label} className={`px-1.5 py-0.5 rounded bg-gray-800/60 ${color}`}>
                    {label}: {val > 0 ? '+' : ''}{(val ?? 0).toFixed(3)}
                  </span>
                ))}
              </div>

              {/* Confidence bar */}
              <div>
                <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                  <span>Confidence</span>
                  <span className="font-mono">{confPct}% · n={p.sample_count}</span>
                </div>
                <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${confPct >= 60 ? 'bg-green-500' : confPct >= 30 ? 'bg-yellow-500' : 'bg-red-500'}`}
                    style={{ width: `${confPct}%` }}
                  />
                </div>
              </div>

              <div className="text-[10px] text-gray-600">
                Last updated {new Date(p.last_updated).toLocaleDateString()}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function buildChartPoint(iteration, score) {
  return {
    iteration,
    overall: score.overall_score,
    language: score.language_roi,
    attention: score.attention_roi,
    visual: score.visual_roi,
  }
}
