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
  const [inputMode, setInputMode] = useState('url')       // 'url' | 'file'
  const [url, setUrl] = useState('')
  const [uploadedHtml, setUploadedHtml] = useState(null)  // { content, filename, screenshot, score, gaze }
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [htmlJobId, setHtmlJobId] = useState(null)
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
  const [currentJobId, setCurrentJobId] = useState(null)
  const [memoryStats, setMemoryStats] = useState(null)
  const [events, setEvents] = useState([])               // raw SSE events for AgentVisionPanel
  const [agentVisionActive, setAgentVisionActive] = useState(false)
  const [intent, setIntent] = useState('engage')
  const [brainRegions, setBrainRegions] = useState(null)
  const [ethicsFlags, setEthicsFlags] = useState([])
  const [intentReward, setIntentReward] = useState(null)
  const [gazeRegions, setGazeRegions] = useState([])
  const [gazeOverlayBase64, setGazeOverlayBase64] = useState('')
  const [analysisView, setAnalysisView] = useState('brain') // brain | deepgaze
  const [backendInfo, setBackendInfo] = useState(null)
  const [optimizedHtml, setOptimizedHtml] = useState(null)  // final HTML string for file mode iframe
  const [pendingApproval, setPendingApproval] = useState(null)

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

  useEffect(() => {
    fetch('/health')
      .then(r => r.ok ? r.json() : null)
      .then(data => data && setBackendInfo(data))
      .catch(() => {})
  }, [])

  const handleEvent = useCallback((type, data) => {
    // Accumulate all raw events for AgentVisionPanel.
    // Keep a single live "scoring_wait" event and update it in place.
    setEvents(prev => {
      const stamped = { type, data, ts: Date.now() }
      if (type === 'progress' && data?.status === 'scoring_wait') {
        const last = prev[prev.length - 1]
        if (last?.type === 'progress' && last?.data?.status === 'scoring_wait') {
          return [...prev.slice(0, -1), { ...stamped, ts: last.ts ?? stamped.ts }]
        }
      }
      return [...prev, stamped]
    })
    if (type === 'progress') {
      const { status: s } = data

      if (s === 'baseline' && data.score) {
        setBaselineScore(data.score)
        setChartData([buildChartPoint(0, data.score)])
        setFeed(prev => [...prev, { kind: 'baseline', score: data.score }])
        return
      }

      if (s === 'iteration_complete') {
        setPendingApproval(null)
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
          setAcceptedEdits(prev => [...prev, { ...data.edit, iteration: data.iteration_count }])
        }
        return
      }

      // Generic status messages (scraping / scoring / proposing)
      if (['scraping', 'scoring', 'scoring_wait', 'proposing'].includes(s)) {
        setFeed(prev => {
          const last = prev[prev.length - 1]
          if (s === 'scoring_wait') {
            const nextItem = {
              kind: 'status',
              status: 'scoring_wait',
              message: data.message,
              wait_started_at: data.wait_started_at,
              timeout_seconds: data.timeout_seconds,
            }
            if (last?.kind === 'status' && last.status === 'scoring_wait') {
              return [...prev.slice(0, -1), nextItem]
            }
            return [...prev, nextItem]
          }
          if (last?.kind === 'status' && last.status === s && last.message === data.message) return prev
          return [...prev, { kind: 'status', status: s, message: data.message }]
        })
      }
      if (s === 'approval_needed') {
        setPendingApproval(data)
      }
    }

    if (type === 'brain_regions') {
      setBrainRegions(data.regions)
      setEthicsFlags(data.ethics_flags || [])
      if (data.intent_reward != null) setIntentReward(data.intent_reward)
    }

    if (type === 'gaze') {
      setGazeRegions(data.gaze_regions || [])
      if (data.gaze_overlay_base64) {
        setGazeOverlayBase64(data.gaze_overlay_base64)
      }
    }

    if (type === 'complete') {
      setPendingApproval(null)
      setCurrentJobId(null)
      setStatus('complete')
      setFinalScore(data.final_score)
      if (data.accepted_edits) setAcceptedEdits(data.accepted_edits)
      if (data.final_brain_regions) setBrainRegions(data.final_brain_regions)
      if (data.ethics_flags) setEthicsFlags(data.ethics_flags)
      setJobIdForPreview(data.job_id)
      if (data.optimized_html) setOptimizedHtml(data.optimized_html)
      esRef.current?.close()
    }

    if (type === 'progress' && data.status === 'rendering') {
      setFeed(prev => [...prev, { kind: 'status', status: 'rendering', message: data.message }])
    }

    if (type === 'error') {
      setPendingApproval(null)
      setCurrentJobId(null)
      setStatus('error')
      setError(data.message)
      esRef.current?.close()
    }
  }, [])

  const handleFileUpload = async (file) => {
    if (!file || !file.name.endsWith('.html')) {
      setError('Please upload a .html file')
      return
    }
    setUploading(true)
    setError(null)
    setUploadedHtml(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API}/upload-html`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`Upload failed: HTTP ${res.status}`)
      const data = await res.json()
      setUploadedHtml(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileUpload(file)
  }

  const startOptimization = async () => {
    if (inputMode === 'file') {
      if (!uploadedHtml) return
      await startHtmlOptimization()
      return
    }
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
    setCurrentJobId(null)
    setPendingApproval(null)
    setPreviewTab('after')
    setBrainRegions(null)
    setEthicsFlags([])
    setIntentReward(null)
    setOptimizedHtml(null)
    setGazeRegions([])
    setGazeOverlayBase64('')
    setAnalysisView('brain')

    try {
      const res = await fetch(`${API}/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim(), max_iterations: maxIter, intent }),
      })
      if (!res.ok) throw new Error(`Server error: HTTP ${res.status}`)
      const { job_id } = await res.json()
      setCurrentJobId(job_id)

      setStatus('running')
      const es = new EventSource(`${API}/job/${job_id}/stream`)
      esRef.current = es

      es.onmessage = (e) => {
        const { type, data } = JSON.parse(e.data)
        handleEvent(type, data)
      }
      attachResilientErrorHandler(es, () => {
        setStatus('error')
        setError('Connection to server lost. Is the backend running?')
      })
    } catch (err) {
      setError(err.message)
      setStatus('idle')
    }
  }

  const startHtmlOptimization = async () => {
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
    setCurrentJobId(null)
    setPendingApproval(null)
    setHtmlJobId(null)
    setGazeRegions([])
    setGazeOverlayBase64('')
    setAnalysisView('brain')

    try {
      const res = await fetch(`${API}/optimize-html`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          html_content: uploadedHtml.html_content,
          filename: uploadedHtml.filename,
          max_iterations: maxIter,
        }),
      })
      if (!res.ok) throw new Error(`Server error: HTTP ${res.status}`)
      const { job_id } = await res.json()

      setHtmlJobId(job_id)
      setCurrentJobId(job_id)
      setStatus('running')
      const es = new EventSource(`${API}/job/${job_id}/stream`)
      esRef.current = es

      es.onmessage = (e) => {
        const { type, data } = JSON.parse(e.data)
        handleEvent(type, data)
        if (type === 'complete') setJobIdForPreview(job_id)
      }
      attachResilientErrorHandler(es, () => {
        setStatus('error')
        setError('Connection to server lost. Is the backend running?')
      })
    } catch (err) {
      setError(err.message)
      setStatus('idle')
    }
  }

  const downloadOptimizedHtml = () => {
    if (!htmlJobId) return
    const a = document.createElement('a')
    a.href = `${API}/html-job/${htmlJobId}/download`
    a.download = uploadedHtml
      ? `${uploadedHtml.filename.replace('.html', '')}_neurallens.html`
      : 'optimized_neurallens.html'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const submitApprovalDecision = async (accept) => {
    if (!currentJobId || !pendingApproval?.iteration_count) return
    try {
      await fetch(`${API}/job/${currentJobId}/decision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          iteration: pendingApproval.iteration_count,
          accept,
        }),
      })
      setPendingApproval(null)
    } catch (err) {
      setError(err.message || 'Failed to submit decision')
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
          {backendInfo && (
            <div className="text-[11px] text-gray-500 mb-1 text-right">
              TRIBE: <span className={backendInfo.tribe_live ? 'text-green-400' : 'text-gray-400'}>{backendInfo.tribe_live ? 'live' : 'stub'}</span>{' '}
              · Agent: <span className={backendInfo.openai_live ? 'text-green-400' : 'text-yellow-400'}>{backendInfo.openai_live ? (backendInfo.model || 'live') : 'stub'}</span>
            </div>
          )}
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

      {activeTab === 'optimize' && status === 'running' && pendingApproval && (
        <ApprovalBar
          pending={pendingApproval}
          onDecision={submitApprovalDecision}
        />
      )}

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
          {/* Compact bar at top when vision is active */}
          <div className="flex items-center gap-3 px-4 py-2.5 border-b border-gray-800 flex-shrink-0 bg-gray-950/80">
            {inputMode === 'url' ? (
              <input
                type="url"
                value={url}
                onChange={e => setUrl(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !isRunning && startOptimization()}
                placeholder="https://example.com"
                disabled={isRunning}
                className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 disabled:opacity-50 transition"
              />
            ) : (
              <span className={`flex-1 text-xs px-3 py-1.5 rounded-lg border truncate ${
                uploadedHtml ? 'border-green-700/60 text-green-300 bg-green-950/20' : 'border-gray-700 text-gray-500'
              }`}>
                {uploadedHtml ? `📄 ${uploadedHtml.filename}` : 'No HTML file uploaded'}
              </span>
            )}
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
              disabled={(inputMode === 'url' ? !url.trim() : !uploadedHtml) || isRunning}
              className="bg-violet-600 hover:bg-violet-500 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold py-1.5 px-4 rounded-lg text-sm transition cursor-pointer disabled:cursor-not-allowed flex-shrink-0"
            >
              {status === 'starting' ? 'Starting…' : isRunning ? 'Running…'
                : inputMode === 'file' ? '⚡ Optimize HTML' : '⚡ Optimize'}
            </button>
            {error && (
              <div className="text-xs text-red-400 truncate max-w-xs">{error}</div>
            )}
          </div>
          {/* After file-mode job completes: swap to live before/after iframe comparison */}
          {status === 'complete' && inputMode === 'file' && uploadedHtml && optimizedHtml ? (
            <div className="flex-1 min-h-0 overflow-y-auto p-4">
              <HtmlBeforeAfterPanel
                originalHtml={uploadedHtml.html_content}
                optimizedHtml={optimizedHtml}
                filename={uploadedHtml.filename}
                baselineScore={baselineScore}
                finalScore={finalScore}
                onDownload={downloadOptimizedHtml}
              />
            </div>
          ) : (
            /* Full-height Agent Vision Panel while running */
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
          )}
        </div>
      )}

      {/* ── Optimize tab — normal split layout ── */}
      {activeTab !== 'build' && activeTab !== 'patterns' && !agentVisionActive && (
      <div className="flex flex-1 min-h-0">
        {/* ── Left panel ── */}
        <div className="w-[40%] border-r border-gray-800 flex flex-col p-5 gap-4 min-h-0">
          {/* Input mode toggle + inputs */}
          <div className="flex flex-col gap-2.5 flex-shrink-0">
            {/* Mode toggle */}
            <div className="flex rounded-lg overflow-hidden border border-gray-700 text-xs w-fit">
              {[['url', '🔗 URL'], ['file', '📄 HTML File']].map(([mode, label]) => (
                <button
                  key={mode}
                  onClick={() => { setInputMode(mode); setError(null) }}
                  disabled={isRunning}
                  className={`px-3 py-1.5 font-medium transition cursor-pointer disabled:cursor-not-allowed ${
                    inputMode === mode
                      ? 'bg-violet-600 text-white'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {inputMode === 'url' ? (
              <input
                type="url"
                value={url}
                onChange={e => setUrl(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !isRunning && startOptimization()}
                placeholder="https://example.com"
                disabled={isRunning}
                className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3.5 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 disabled:opacity-50 transition"
              />
            ) : (
              /* File upload zone */
              <div
                onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                className={`relative flex flex-col items-center justify-center gap-2 border-2 border-dashed rounded-xl px-4 py-5 text-center transition cursor-pointer ${
                  isDragging
                    ? 'border-violet-500 bg-violet-950/30'
                    : uploadedHtml
                    ? 'border-green-700/60 bg-green-950/20'
                    : 'border-gray-700 hover:border-gray-500 bg-gray-900/40'
                }`}
                onClick={() => {
                  const inp = document.createElement('input')
                  inp.type = 'file'
                  inp.accept = '.html'
                  inp.onchange = e => e.target.files[0] && handleFileUpload(e.target.files[0])
                  inp.click()
                }}
              >
                {uploading ? (
                  <span className="text-xs text-gray-400 flex items-center gap-2">
                    <span className="animate-spin">⟳</span> Rendering HTML…
                  </span>
                ) : uploadedHtml ? (
                  <>
                    <span className="text-green-400 text-lg">✓</span>
                    <span className="text-xs text-green-300 font-medium">{uploadedHtml.filename}</span>
                    <span className="text-[10px] text-gray-500">
                      Baseline: {uploadedHtml.page_score?.overall_score?.toFixed(4) ?? '—'}
                    </span>
                    {uploadedHtml.screenshot_base64 && (
                      <img
                        src={`data:image/png;base64,${uploadedHtml.screenshot_base64}`}
                        alt="Uploaded page preview"
                        className="mt-1 w-full rounded border border-gray-700/50 object-cover max-h-28"
                      />
                    )}
                    <span className="text-[10px] text-gray-600 mt-1">Click to replace</span>
                  </>
                ) : (
                  <>
                    <span className="text-2xl">📄</span>
                    <span className="text-xs text-gray-400 font-medium">
                      Drop your <span className="text-violet-400">.html</span> file here
                    </span>
                    <span className="text-[10px] text-gray-600">or click to browse</span>
                  </>
                )}
              </div>
            )}

            <div className="flex gap-2 items-center flex-wrap">
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
              {inputMode === 'url' && (
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
              )}
              <button
                onClick={startOptimization}
                disabled={(inputMode === 'url' ? !url.trim() : !uploadedHtml) || isRunning}
                className="flex-1 bg-violet-600 hover:bg-violet-500 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold py-2.5 px-4 rounded-lg text-sm transition cursor-pointer disabled:cursor-not-allowed"
              >
                {status === 'starting' ? 'Starting…' : isRunning ? 'Running…'
                  : inputMode === 'file' ? '⚡ Optimize HTML' : '⚡ Optimize'}
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
                  {inputMode === 'file'
                    ? 'Upload an HTML file and click Optimize HTML to begin'
                    : 'Enter a URL and click Optimize to begin neural analysis'}
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

          {/* Analysis panels: Brain vs DeepGaze */}
          {(brainRegions || gazeRegions.length > 0 || status === 'running' || status === 'complete') && (
            <div className="flex-shrink-0 bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
                  Neural Analysis
                </h3>
                <div className="flex rounded-lg overflow-hidden border border-gray-700 text-xs">
                  <button
                    onClick={() => setAnalysisView('brain')}
                    className={`px-3 py-1.5 font-medium transition ${
                      analysisView === 'brain'
                        ? 'bg-violet-600 text-white'
                        : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                    }`}
                  >
                    Brain
                  </button>
                  <button
                    onClick={() => setAnalysisView('deepgaze')}
                    className={`px-3 py-1.5 font-medium transition ${
                      analysisView === 'deepgaze'
                        ? 'bg-violet-600 text-white'
                        : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                    }`}
                  >
                    DeepGaze
                  </button>
                </div>
              </div>
              {analysisView === 'brain' ? (
                <BrainPanel
                  regions={brainRegions}
                  ethicsFlags={ethicsFlags}
                  intent={intent}
                  intentReward={intentReward}
                />
              ) : (
                <DeepGazePanel
                  regions={gazeRegions}
                  overlayBase64={gazeOverlayBase64}
                />
              )}
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
              <div className="flex flex-col gap-3 max-h-[520px] overflow-y-auto pr-1">
                {acceptedEdits.map((edit, idx) => (
                  <DiffCard
                    key={idx}
                    edit={edit}
                    index={idx}
                    jobId={currentJobId || jobIdForPreview}
                  />
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

          {/* HTML file before/after iframe comparison */}
          {status === 'complete' && inputMode === 'file' && uploadedHtml && optimizedHtml && (
            <HtmlBeforeAfterPanel
              originalHtml={uploadedHtml.html_content}
              optimizedHtml={optimizedHtml}
              filename={uploadedHtml.filename}
              baselineScore={baselineScore}
              finalScore={finalScore}
              onDownload={downloadOptimizedHtml}
            />
          )}

          {/* Before / After page preview */}
          {status === 'complete' && jobIdForPreview && (
            <PreviewPanel
              jobId={jobIdForPreview}
              tab={previewTab}
              onTab={setPreviewTab}
              baselineScore={baselineScore}
              finalScore={finalScore}
              acceptedEdits={acceptedEdits}
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
                    ['🤖', 'AI agent proposes targeted text edits based on gaze + brain data'],
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

function HtmlBeforeAfterPanel({ originalHtml, optimizedHtml, filename, baselineScore, finalScore, onDownload }) {
  const [activePane, setActivePane] = useState('split') // 'split' | 'before' | 'after'

  const improvement = baselineScore && finalScore
    ? (((finalScore.overall_score - baselineScore.overall_score) / Math.max(baselineScore.overall_score, 1e-6)) * 100).toFixed(1)
    : null

  return (
    <div className="flex flex-col gap-3 flex-shrink-0">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-gray-300">Live HTML Comparison</span>
          <span className="text-[10px] text-gray-600 truncate max-w-[180px]">{filename}</span>
          {improvement !== null && (
            <span className={`text-[11px] font-bold px-2 py-0.5 rounded ${
              parseFloat(improvement) >= 0 ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'
            }`}>
              {parseFloat(improvement) >= 0 ? '+' : ''}{improvement}% neural score
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg overflow-hidden border border-gray-700 text-[11px]">
            {[['split','⬛ Split'],['before','📄 Original'],['after','✨ Optimized']].map(([v, label]) => (
              <button key={v} onClick={() => setActivePane(v)}
                className={`px-3 py-1.5 font-medium transition cursor-pointer ${
                  activePane === v ? (v === 'after' ? 'bg-violet-600 text-white' : 'bg-gray-700 text-white') : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`}>{label}</button>
            ))}
          </div>
          <button onClick={onDownload}
            className="flex items-center gap-1.5 bg-violet-600 hover:bg-violet-500 text-white font-semibold py-1.5 px-3 rounded-lg text-[11px] transition cursor-pointer">
            ⬇ Download
          </button>
        </div>
      </div>

      {/* Iframe panes */}
      <div className={`rounded-xl overflow-hidden border border-gray-700/50 bg-gray-950 ${activePane === 'split' ? 'flex gap-0' : ''}`} style={{ height: 540 }}>
        {activePane === 'split' ? (
          <>
            <div className="flex-1 relative border-r border-gray-700/50">
              <div className="absolute top-2 left-2 z-10 text-[10px] font-bold px-2 py-0.5 rounded bg-gray-800/90 text-gray-400 border border-gray-700">ORIGINAL</div>
              <iframe srcDoc={originalHtml} className="w-full h-full border-0 bg-white" sandbox="allow-same-origin" title="Original HTML" />
            </div>
            <div className="flex-1 relative">
              <div className="absolute top-2 left-2 z-10 text-[10px] font-bold px-2 py-0.5 rounded bg-violet-800/90 text-violet-200 border border-violet-700">NEURAL-OPTIMIZED</div>
              <iframe srcDoc={optimizedHtml} className="w-full h-full border-0 bg-white" sandbox="allow-same-origin" title="Optimized HTML" />
            </div>
          </>
        ) : activePane === 'before' ? (
          <div className="relative w-full h-full">
            <div className="absolute top-2 left-2 z-10 text-[10px] font-bold px-2 py-0.5 rounded bg-gray-800/90 text-gray-400 border border-gray-700">ORIGINAL</div>
            <iframe srcDoc={originalHtml} className="w-full h-full border-0 bg-white" sandbox="allow-same-origin" title="Original HTML" />
          </div>
        ) : (
          <div className="relative w-full h-full">
            <div className="absolute top-2 left-2 z-10 text-[10px] font-bold px-2 py-0.5 rounded bg-violet-800/90 text-violet-200 border border-violet-700">NEURAL-OPTIMIZED</div>
            <iframe srcDoc={optimizedHtml} className="w-full h-full border-0 bg-white" sandbox="allow-same-origin" title="Optimized HTML" />
          </div>
        )}
      </div>

      {activePane === 'split' && (
        <p className="text-[11px] text-gray-600 text-center">
          Original on the left · NeuralLens-optimized on the right · Both are live rendered HTML
        </p>
      )}
    </div>
  )
}

function ApprovalBar({ pending, onDecision }) {
  const [expanded, setExpanded] = useState(true)
  const edit = pending.edit || {}
  const delta = Number(pending.score_delta ?? 0)
  const positive = delta >= 0
  const roiDeltas = pending.roi_deltas || {}
  const expectedROI = edit.expected_roi_impact || {}
  const actionType = edit.action_type || 'edit'
  const actionIcon = ACTION_ICONS[actionType] || '🔧'
  const truncate = (s, n = 220) => (s && s.length > n ? s.slice(0, n) + '…' : s)
  const original = edit.original ?? edit.html_original ?? ''
  const replacement = edit.replacement ?? edit.html_replacement ?? ''

  return (
    <div className="px-4 py-3 border-b-2 border-violet-700/60 bg-gradient-to-r from-violet-950/60 via-gray-950/80 to-cyan-950/40 sticky top-0 z-30 backdrop-blur">
      <div className="max-w-6xl mx-auto flex flex-col gap-2">
        {/* Top row */}
        <div className="flex items-center gap-3 flex-wrap">
          <span className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-violet-300">
            <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
            Run paused — your decision required
          </span>
          <span className="text-[10px] text-gray-500 px-2 py-0.5 rounded-full border border-gray-700">
            Iter {pending.iteration_count}/{pending.max_iterations}
          </span>
          <span className="text-xs text-gray-300 flex items-center gap-1.5">
            <span>{actionIcon}</span>
            <span className="capitalize font-medium">{actionType.replace(/_/g, ' ')}</span>
          </span>
          <span className="text-xs text-gray-400 font-mono">
            {Number(pending.current_overall ?? 0).toFixed(4)}
            {' → '}
            <span className={positive ? 'text-green-400' : 'text-red-400'}>
              {Number(pending.proposed_overall ?? 0).toFixed(4)}
              {' '}({positive ? '+' : ''}{delta.toFixed(4)})
            </span>
          </span>
          <button
            onClick={() => setExpanded(v => !v)}
            className="text-[10px] text-gray-500 hover:text-gray-300 underline cursor-pointer"
          >
            {expanded ? 'Hide details' : 'Show details'}
          </button>
          <div className="ml-auto flex gap-2">
            <button
              onClick={() => onDecision(false)}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg border border-red-700/60 text-red-300 hover:bg-red-900/40 transition cursor-pointer"
            >
              ✗ Reject
            </button>
            <button
              onClick={() => onDecision(true)}
              className="px-4 py-1.5 text-xs font-semibold rounded-lg bg-green-600/80 hover:bg-green-500 text-white transition cursor-pointer shadow-md shadow-green-900/40"
            >
              ✓ Accept
            </button>
          </div>
        </div>

        {/* Details */}
        {expanded && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 text-xs">
            {/* Diff */}
            <div className="lg:col-span-2 flex flex-col gap-1.5">
              {edit.target && (
                <div className="text-[10px] text-gray-500">
                  Target: <span className="text-gray-300">{edit.target}</span>
                  {edit.html_selector && (
                    <span className="ml-2 font-mono text-violet-400">{edit.html_selector}</span>
                  )}
                </div>
              )}
              {original && (
                <div className="line-through text-red-400/80 bg-red-950/30 border border-red-900/40 px-2.5 py-2 rounded leading-relaxed break-words">
                  {truncate(original)}
                </div>
              )}
              {replacement && (
                <div className="text-green-300 bg-green-950/30 border border-green-900/40 px-2.5 py-2 rounded leading-relaxed break-words">
                  {truncate(replacement)}
                </div>
              )}
              {edit.reasoning && (
                <div className="text-gray-400 italic leading-relaxed pt-1">
                  💭 {truncate(edit.reasoning, 320)}
                </div>
              )}
            </div>

            {/* Score breakdown */}
            <div className="flex flex-col gap-2">
              <div className="bg-gray-900/60 border border-gray-800 rounded-lg p-2.5">
                <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-1">
                  ROI deltas (predicted vs measured)
                </div>
                <div className="grid grid-cols-3 gap-1.5 font-mono text-[11px]">
                  {['language', 'attention', 'visual'].map(k => {
                    const measured = Number(roiDeltas[k] ?? 0)
                    const predicted = Number(expectedROI[k] ?? 0)
                    return (
                      <div key={k} className="bg-gray-950 rounded px-1.5 py-1">
                        <div className="text-[9px] text-gray-500 capitalize">{k}</div>
                        <div className={measured >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {measured >= 0 ? '+' : ''}{measured.toFixed(3)}
                        </div>
                        <div className="text-[9px] text-gray-600">
                          pred {predicted >= 0 ? '+' : ''}{predicted.toFixed(2)}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
              {pending.default_decision && (
                <div className="text-[10px] text-gray-500">
                  Auto-suggestion: <span className={
                    pending.default_decision === 'accept' ? 'text-green-400' : 'text-red-400'
                  }>{pending.default_decision}</span>
                  {' '}(based on score delta)
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

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
    if (item.status === 'scoring_wait') {
      return <ScoringWaitStatus item={item} />
    }
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

function ScoringWaitStatus({ item }) {
  const [nowSec, setNowSec] = useState(() => Math.floor(Date.now() / 1000))
  useEffect(() => {
    const id = setInterval(() => setNowSec(Math.floor(Date.now() / 1000)), 1000)
    return () => clearInterval(id)
  }, [])

  const startedAt = Number(item.wait_started_at || 0)
  const timeout = Math.max(1, Number(item.timeout_seconds || 90))
  const elapsed = startedAt > 0 ? Math.max(0, nowSec - Math.floor(startedAt)) : 0
  const remaining = Math.max(0, timeout - elapsed)

  return (
    <div className="px-3 py-1.5 text-xs text-gray-500 flex items-center gap-2">
      <span className="animate-spin inline-block">⟳</span>
      <span>
        {item.message} Timeout in ~{remaining}s (elapsed {elapsed}s).
      </span>
    </div>
  )
}

function DiffCard({ edit, index, jobId }) {
  const maxLen = 140
  const truncate = (s) => s && s.length > maxLen ? s.slice(0, maxLen) + '…' : s
  const [thumbError, setThumbError] = useState(false)
  const [showThumb, setShowThumb] = useState(true)
  const iter = edit.iteration

  const thumbSrc = (jobId && iter)
    ? `/job/${jobId}/iteration/${iter}/screenshot`
    : null

  return (
    <div className="text-xs bg-gray-800/50 rounded-lg p-3 border border-gray-700/40">
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <span className="text-gray-500 font-mono">{index + 1}.</span>
        <span className="capitalize text-gray-200 font-semibold">
          {(edit.action_type || '').replace(/_/g, ' ')}
        </span>
        {iter != null && (
          <span className="text-[10px] font-mono text-gray-500 bg-gray-900/60 px-1.5 py-0.5 rounded border border-gray-700/40">
            iter {iter}
          </span>
        )}
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

      <div className="grid grid-cols-1 md:grid-cols-[1fr_180px] gap-3">
        <div className="min-w-0">
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

        {/* Iteration thumbnail — visual proof of the rendered page state */}
        {thumbSrc && showThumb && !thumbError && (
          <div className="relative group rounded overflow-hidden border border-violet-900/30 bg-gray-950">
            <a href={thumbSrc} target="_blank" rel="noreferrer" title="Open full screenshot">
              <img
                src={thumbSrc}
                alt={`Iteration ${iter} render`}
                className="block w-full h-[180px] object-cover object-top"
                onError={() => setThumbError(true)}
              />
            </a>
            <div className="absolute top-1 left-1 pointer-events-none">
              <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-violet-800/90 text-violet-100 border border-violet-700">
                AFTER iter {iter}
              </span>
            </div>
            <button
              onClick={() => setShowThumb(false)}
              className="absolute top-1 right-1 text-[9px] px-1.5 py-0.5 rounded bg-black/70 text-gray-300 border border-gray-700 opacity-0 group-hover:opacity-100 transition cursor-pointer"
              title="Hide thumbnail"
            >
              ×
            </button>
          </div>
        )}
      </div>
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
  const v = Math.max(0, Math.min(1, Number(value) || 0))
  return (
    <div className="text-center">
      <div className={`text-3xl font-bold ${highlight ? 'text-violet-300' : 'text-gray-300'}`}>
        {v.toFixed(3)}
      </div>
      <div className="text-xs text-gray-400 mt-1">{label}</div>
      <div className="text-[10px] text-gray-600">normalized 0-1</div>
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

function PreviewPanel({ jobId, tab, onTab, baselineScore, finalScore, acceptedEdits = [] }) {
  // Modes:
  //   "split"   – side-by-side, synchronized scroll (default, the most useful)
  //   "slider"  – classic before/after photo slider with draggable divider
  //   "toggle"  – legacy single-image tab (Before | After)
  const [mode, setMode] = useState('split')
  const [imgError, setImgError] = useState({ before: false, after: false })
  const [imgLoaded, setImgLoaded] = useState({ before: false, after: false })
  const [sliderPct, setSliderPct] = useState(50)
  const beforeBoxRef = useRef(null)
  const afterBoxRef = useRef(null)
  const sliderBoxRef = useRef(null)
  const isDraggingRef = useRef(false)

  const beforeSrc = `/job/${jobId}/before-screenshot`
  const afterSrc  = `/job/${jobId}/after-screenshot`

  const overallDelta = (finalScore?.overall_score ?? 0) - (baselineScore?.overall_score ?? 0)
  const pct = baselineScore?.overall_score
    ? (overallDelta / baselineScore.overall_score) * 100
    : 0

  // Synchronized scroll for split mode — scrolling either pane scrolls the other
  // proportionally (handles different image heights gracefully).
  useEffect(() => {
    if (mode !== 'split') return
    const a = beforeBoxRef.current
    const b = afterBoxRef.current
    if (!a || !b) return
    let lock = false
    const handler = (src, dst) => () => {
      if (lock) return
      lock = true
      const ratio = src.scrollTop / Math.max(src.scrollHeight - src.clientHeight, 1)
      dst.scrollTop = ratio * Math.max(dst.scrollHeight - dst.clientHeight, 1)
      requestAnimationFrame(() => { lock = false })
    }
    const onA = handler(a, b)
    const onB = handler(b, a)
    a.addEventListener('scroll', onA, { passive: true })
    b.addEventListener('scroll', onB, { passive: true })
    return () => {
      a.removeEventListener('scroll', onA)
      b.removeEventListener('scroll', onB)
    }
  }, [mode, imgLoaded.before, imgLoaded.after])

  // Slider drag — pointer-anywhere-on-strip semantics, with global mouseup.
  useEffect(() => {
    if (mode !== 'slider') return
    const move = (e) => {
      if (!isDraggingRef.current) return
      const box = sliderBoxRef.current
      if (!box) return
      const rect = box.getBoundingClientRect()
      const x = (e.touches?.[0]?.clientX ?? e.clientX) - rect.left
      const ratio = Math.min(Math.max(x / rect.width, 0), 1)
      setSliderPct(ratio * 100)
    }
    const up = () => { isDraggingRef.current = false }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    window.addEventListener('touchmove', move, { passive: false })
    window.addEventListener('touchend', up)
    return () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
      window.removeEventListener('touchmove', move)
      window.removeEventListener('touchend', up)
    }
  }, [mode])

  const onBeforeImgLoad = () => setImgLoaded(s => ({ ...s, before: true }))
  const onAfterImgLoad  = () => setImgLoaded(s => ({ ...s, after: true }))

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex-shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
            Before / After
          </h3>
          {finalScore && baselineScore && (
            <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
              overallDelta >= 0
                ? 'bg-green-900/30 border-green-700/40 text-green-300'
                : 'bg-red-900/30 border-red-700/40 text-red-300'
            }`}>
              overall {baselineScore.overall_score?.toFixed(4)} → {finalScore.overall_score?.toFixed(4)}
              {' '}({overallDelta >= 0 ? '+' : ''}{pct.toFixed(1)}%)
            </span>
          )}
          {acceptedEdits.length > 0 && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded border bg-violet-900/30 border-violet-700/40 text-violet-300">
              {acceptedEdits.length} accepted edit{acceptedEdits.length === 1 ? '' : 's'} applied
            </span>
          )}
        </div>

        {/* Mode switcher */}
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg overflow-hidden border border-gray-700 text-[10px]">
            {[
              { id: 'split',  label: '⬛⬜ Split' },
              { id: 'slider', label: '⇆ Slider' },
              { id: 'toggle', label: '⇄ Toggle' },
            ].map(opt => (
              <button
                key={opt.id}
                onClick={() => setMode(opt.id)}
                className={`px-2.5 py-1 font-medium transition ${
                  mode === opt.id
                    ? 'bg-violet-600 text-white'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* SPLIT MODE — side-by-side with synced scroll */}
      {mode === 'split' && (
        <div className="grid grid-cols-2 gap-3">
          {[
            { side: 'before', src: beforeSrc, ref: beforeBoxRef, label: 'BEFORE — original', badge: 'bg-gray-800/90 text-gray-300 border-gray-700', onLoad: onBeforeImgLoad },
            { side: 'after',  src: afterSrc,  ref: afterBoxRef,  label: 'AFTER — neural-optimized', badge: 'bg-violet-800/90 text-violet-100 border-violet-700', onLoad: onAfterImgLoad },
          ].map(({ side, src, ref, label, badge, onLoad }) => (
            <div key={side} className="relative">
              <div
                ref={ref}
                className="rounded-lg overflow-y-auto overflow-x-hidden border border-gray-700/50 bg-gray-950"
                style={{ height: '560px' }}
              >
                {!imgLoaded[side] && !imgError[side] && (
                  <div className="flex items-center justify-center h-full gap-3 text-gray-500 text-sm">
                    <span className="animate-spin">⟳</span> Loading {side}…
                  </div>
                )}
                {imgError[side] && (
                  <div className="flex items-center justify-center h-full text-gray-600 text-sm">
                    Screenshot not available
                  </div>
                )}
                <img
                  src={src}
                  alt={label}
                  className={`w-full block ${imgLoaded[side] ? '' : 'hidden'}`}
                  onLoad={onLoad}
                  onError={() => setImgError(s => ({ ...s, [side]: true }))}
                />
              </div>
              <div className="absolute top-2 left-2 pointer-events-none">
                <span className={`text-[10px] font-semibold px-2 py-1 rounded border ${badge}`}>
                  {label}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* SLIDER MODE — classic before/after with draggable vertical divider.
          Both images are stacked at the same width; the AFTER image is
          clipped via clip-path to reveal the BEFORE on the right of the
          handle. This guarantees pixel-perfect alignment regardless of
          their natural sizes. */}
      {mode === 'slider' && (
        <div
          ref={sliderBoxRef}
          className="relative rounded-lg overflow-hidden border border-gray-700/50 bg-gray-950 cursor-ew-resize select-none"
          style={{ height: '560px' }}
          onMouseDown={(e) => { isDraggingRef.current = true; e.preventDefault() }}
          onTouchStart={() => { isDraggingRef.current = true }}
        >
          {/* Single scroll container; the BEFORE image sets the layout, the
              AFTER image is absolutely positioned on top at the same size. */}
          <div className="absolute inset-0 overflow-y-auto overflow-x-hidden">
            <div className="relative w-full">
              <img
                src={beforeSrc}
                alt="Before"
                className="w-full block"
                onLoad={onBeforeImgLoad}
                onError={() => setImgError(s => ({ ...s, before: true }))}
                draggable={false}
              />
              <img
                src={afterSrc}
                alt="After"
                className="absolute inset-0 w-full block"
                style={{
                  clipPath: `inset(0 ${100 - sliderPct}% 0 0)`,
                  WebkitClipPath: `inset(0 ${100 - sliderPct}% 0 0)`,
                }}
                onLoad={onAfterImgLoad}
                onError={() => setImgError(s => ({ ...s, after: true }))}
                draggable={false}
              />
            </div>
          </div>

          {/* Divider line + handle (sticky to the viewport so they're always visible) */}
          <div
            className="absolute inset-y-0 pointer-events-none"
            style={{ left: `calc(${sliderPct}% - 1px)`, width: '2px', background: 'rgba(167, 139, 250, 0.85)', boxShadow: '0 0 12px rgba(167, 139, 250, 0.5)' }}
          />
          <div
            className="absolute pointer-events-none"
            style={{
              left: `calc(${sliderPct}% - 16px)`,
              top: '50%',
              transform: 'translateY(-50%)',
              width: '32px',
              height: '32px',
              borderRadius: '9999px',
              background: 'rgba(167, 139, 250, 0.95)',
              boxShadow: '0 0 0 4px rgba(15,23,42,0.7), 0 4px 16px rgba(0,0,0,0.6)',
              display: 'grid',
              placeItems: 'center',
              color: 'white',
              fontSize: '13px',
              fontWeight: 700,
            }}
          >
            ⇆
          </div>
          <div className="absolute top-2 left-2 pointer-events-none">
            <span className="text-[10px] font-semibold px-2 py-1 rounded bg-violet-800/90 text-violet-100 border border-violet-700">
              ✨ AFTER
            </span>
          </div>
          <div className="absolute top-2 right-2 pointer-events-none">
            <span className="text-[10px] font-semibold px-2 py-1 rounded bg-gray-800/90 text-gray-300 border border-gray-700">
              📄 BEFORE
            </span>
          </div>
        </div>
      )}

      {/* TOGGLE MODE — single image tab (legacy) */}
      {mode === 'toggle' && (
        <div>
          <div className="flex justify-end mb-2">
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
          <div className="relative rounded-lg overflow-hidden border border-gray-700/50 bg-gray-950" style={{ height: '560px' }}>
            <div className={`absolute inset-0 overflow-y-auto transition-opacity duration-300 ${tab === 'before' ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
              <img src={beforeSrc} alt="Original" className="w-full block" onLoad={onBeforeImgLoad}
                   onError={() => setImgError(s => ({ ...s, before: true }))} />
            </div>
            <div className={`absolute inset-0 overflow-y-auto transition-opacity duration-300 ${tab === 'after' ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
              <img src={afterSrc} alt="Optimized" className="w-full block" onLoad={onAfterImgLoad}
                   onError={() => setImgError(s => ({ ...s, after: true }))} />
            </div>
            <div className="absolute top-2 left-2 pointer-events-none">
              {tab === 'before' ? (
                <span className="text-[10px] font-semibold px-2 py-1 rounded bg-gray-800/90 text-gray-300 border border-gray-700">
                  ORIGINAL
                </span>
              ) : (
                <span className="text-[10px] font-semibold px-2 py-1 rounded bg-violet-800/90 text-violet-100 border border-violet-700">
                  NEURAL-OPTIMIZED
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      <p className="text-[11px] text-gray-600 mt-2 text-center">
        {mode === 'split'  && 'Scroll either pane — both stay in sync. The right pane is a fresh Playwright render with all accepted edits applied.'}
        {mode === 'slider' && 'Drag the handle to wipe between original (right) and optimized (left). Replaced text is softly highlighted.'}
        {mode === 'toggle' && 'Switch tabs to compare the original page with the neural-optimized render.'}
      </p>
    </div>
  )
}

function DeepGazePanel({ regions, overlayBase64 }) {
  const top = regions?.[0]
  return (
    <div className="bg-gray-950 border border-gray-800 rounded-xl p-4">
      {!regions?.length ? (
        <div className="text-xs text-gray-500 text-center py-6">
          DeepGaze output appears after gaze analysis completes.
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="text-xs text-gray-400">
            Top fixation region: #{top.rank} at {JSON.stringify(top.peak_coords)} · saliency {top.saliency_score?.toFixed?.(3) ?? top.saliency_score}
          </div>
          {overlayBase64 ? (
            <img
              src={`data:image/png;base64,${overlayBase64}`}
              alt="DeepGaze overlay"
              className="w-full rounded-lg border border-gray-700"
            />
          ) : (
            <div className="text-xs text-gray-500">Heatmap overlay not available yet.</div>
          )}
          <div className="max-h-36 overflow-y-auto border border-gray-800 rounded-lg p-2 bg-gray-900">
            {regions.slice(0, 5).map((r) => (
              <div key={r.rank} className="text-xs text-gray-300 font-mono py-1">
                #{r.rank} sal={r.saliency_score} bbox={JSON.stringify(r.bbox)}
              </div>
            ))}
          </div>
        </div>
      )}
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
  const [refreshing, setRefreshing] = useState(false)

  const loadPatterns = useCallback(async (silent = false) => {
    if (silent) setRefreshing(true)
    try {
      const r = await fetch('/patterns')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const data = await r.json()
      setPatterns(Array.isArray(data) ? data : [])
      setError(null)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
      if (silent) setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    const initialLoad = async () => {
      await loadPatterns(false)
    }
    void initialLoad()

    const intervalId = setInterval(() => {
      if (!active) return
      void loadPatterns(true)
    }, 5000)

    return () => {
      active = false
      clearInterval(intervalId)
    }
  }, [loadPatterns])

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
        <div className="flex items-center gap-2">
          <button
            onClick={() => void loadPatterns(true)}
            className="text-xs bg-gray-900 border border-gray-800 px-2.5 py-1.5 rounded-lg text-gray-300 hover:text-white hover:border-gray-700 transition cursor-pointer"
          >
            {refreshing ? 'Refreshing…' : 'Refresh'}
          </button>
          <span className="text-sm font-mono bg-gray-900 border border-gray-800 px-3 py-1.5 rounded-lg text-violet-400">
            {patterns.length} patterns
          </span>
        </div>
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

// EventSource auto-reconnects on transient drops. We only escalate to the user
// after sustained failures (>15s of CONNECTING/CLOSED with no data).
function attachResilientErrorHandler(es, onFinalError) {
  let timer = null
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) {
      onFinalError?.()
      return
    }
    if (timer) return
    timer = setTimeout(() => {
      if (es.readyState !== EventSource.OPEN) {
        onFinalError?.()
        es.close()
      }
      timer = null
    }, 15000)
  }
  es.addEventListener('open', () => {
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
  })
}

function buildChartPoint(iteration, score) {
  return {
    iteration,
    overall: score.overall_score,
    language: score.language_roi,
    attention: score.attention_roi,
    visual: score.visual_roi,
  }
}
