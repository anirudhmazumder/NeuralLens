/**
 * BrainPanel — real-time 9-region HCP-MMP1 brain activation display.
 *
 * Shows:
 *  - 9 region bars color-coded by category (engagement/trust/penalty/dual)
 *  - Key trio highlighted: Amygdala, Hippocampus, NAcc
 *  - Ethics flags (dark pattern warnings, Yerkes-Dodson ceiling, etc.)
 *  - Intent-aware reward per iteration
 */
import { useMemo } from 'react'

const CATEGORY_COLORS = {
  engagement: '#6366f1',   // indigo
  trust:      '#10b981',   // emerald
  penalty:    '#ef4444',   // red
  dual:       '#f59e0b',   // amber
}

const REGION_CATEGORIES = {
  FFA:         'engagement',
  'V4':        'engagement',
  'MT+':       'engagement',
  Hippocampus: 'engagement',
  PFC:         'trust',
  ACC:         'penalty',
  Amygdala:    'penalty',
  Insula:      'penalty',
  NAcc:        'dual',
}

const REGION_DESCRIPTIONS = {
  FFA:         'Face/social recognition',
  V4:          'Color & visual richness',
  'MT+':       'Motion & dynamic elements',
  Hippocampus: 'Novelty & memorability',
  PFC:         'Clarity & trust',
  ACC:         'Cognitive conflict',
  Amygdala:    'Threat / anxiety signal',
  Insula:      'Visceral unease',
  NAcc:        'Reward anticipation',
}

const SEVERITY_STYLES = {
  block: { bg: '#fef2f2', border: '#fca5a5', text: '#b91c1c', icon: '🚫' },
  warn:  { bg: '#fffbeb', border: '#fcd34d', text: '#92400e', icon: '⚠️' },
  info:  { bg: '#eff6ff', border: '#93c5fd', text: '#1e40af', icon: 'ℹ️' },
}

const KEY_TRIO = ['Amygdala', 'Hippocampus', 'NAcc']

export default function BrainPanel({ regions, ethicsFlags = [], intent = 'engage', intentReward = null }) {
  const sortedRegions = useMemo(() => {
    if (!regions) return []
    return Object.entries(regions).sort((a, b) => b[1] - a[1])
  }, [regions])

  if (!regions) {
    return (
      <div style={{
        background: '#0f172a', border: '1px solid #1e293b',
        borderRadius: 12, padding: '16px', color: '#64748b',
        fontSize: 13, textAlign: 'center',
      }}>
        Brain activation data will appear here once optimization starts.
      </div>
    )
  }

  return (
    <div style={{
      background: '#0f172a', border: '1px solid #1e293b',
      borderRadius: 12, padding: '16px', display: 'flex', flexDirection: 'column', gap: 14,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ color: '#e2e8f0', fontWeight: 700, fontSize: 13, letterSpacing: '0.05em' }}>
          BRAIN ACTIVATION
        </span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{
            background: '#1e293b', color: '#94a3b8', fontSize: 11,
            padding: '2px 8px', borderRadius: 20, fontFamily: 'monospace',
          }}>
            HCP-MMP1 · 9 regions
          </span>
          {intentReward !== null && (
            <span style={{
              background: intentReward >= 0 ? '#052e16' : '#450a0a',
              color: intentReward >= 0 ? '#4ade80' : '#f87171',
              fontSize: 11, padding: '2px 8px', borderRadius: 20, fontFamily: 'monospace',
            }}>
              {intentReward >= 0 ? '+' : ''}{intentReward.toFixed(4)}
            </span>
          )}
        </div>
      </div>

      {/* Key trio: Amygdala, Hippocampus, NAcc */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
        {KEY_TRIO.map(region => {
          const val = regions[region] ?? 0
          const cat = REGION_CATEGORIES[region]
          const color = CATEGORY_COLORS[cat]
          const isWarning = (region === 'Amygdala' && val > 0.60) || (region === 'NAcc' && val > 0.70)
          return (
            <div key={region} style={{
              background: '#1e293b',
              border: `1px solid ${isWarning ? '#f87171' : '#334155'}`,
              borderRadius: 8, padding: '10px 12px', textAlign: 'center',
            }}>
              <div style={{ color: '#94a3b8', fontSize: 10, marginBottom: 4 }}>
                {region}
              </div>
              <div style={{
                fontSize: 22, fontWeight: 800, fontFamily: 'monospace',
                color: isWarning ? '#f87171' : color,
              }}>
                {(val * 100).toFixed(0)}
                <span style={{ fontSize: 11, fontWeight: 400 }}>%</span>
              </div>
              <div style={{ color: '#475569', fontSize: 9, marginTop: 2 }}>
                {REGION_DESCRIPTIONS[region]}
              </div>
            </div>
          )
        })}
      </div>

      {/* All 9 region bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {sortedRegions.map(([region, val]) => {
          const cat = REGION_CATEGORIES[region] || 'engagement'
          const color = CATEGORY_COLORS[cat]
          const pct = Math.round(val * 100)
          return (
            <div key={region}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <span style={{ color: '#cbd5e1', fontSize: 11 }}>
                  {region}
                  <span style={{ color: '#475569', marginLeft: 6, fontSize: 10 }}>
                    {REGION_DESCRIPTIONS[region]}
                  </span>
                </span>
                <span style={{ color, fontSize: 11, fontFamily: 'monospace' }}>
                  {pct}%
                </span>
              </div>
              <div style={{ height: 5, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  width: `${pct}%`, height: '100%',
                  background: color, borderRadius: 3,
                  transition: 'width 0.4s ease',
                }} />
              </div>
            </div>
          )
        })}
      </div>

      {/* Category legend */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
          <div key={cat} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
            <span style={{ color: '#64748b', fontSize: 10 }}>{cat}</span>
          </div>
        ))}
      </div>

      {/* Ethics flags */}
      {ethicsFlags && ethicsFlags.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ color: '#94a3b8', fontSize: 11, fontWeight: 600 }}>ETHICS FLAGS</div>
          {ethicsFlags.map((flag, i) => {
            const style = SEVERITY_STYLES[flag.severity] || SEVERITY_STYLES.info
            return (
              <div key={i} style={{
                background: style.bg, border: `1px solid ${style.border}`,
                borderRadius: 6, padding: '7px 10px',
                display: 'flex', gap: 8, alignItems: 'flex-start',
              }}>
                <span style={{ fontSize: 13, flexShrink: 0 }}>{style.icon}</span>
                <div>
                  <div style={{ color: style.text, fontSize: 11, fontWeight: 600 }}>
                    {flag.code.replace(/_/g, ' ')}
                  </div>
                  <div style={{ color: style.text, fontSize: 10, opacity: 0.85, marginTop: 2 }}>
                    {flag.message}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
