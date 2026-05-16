import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { analyzeBug, fixBug, getUserTasks, healthCheck } from '../api'
import StatsCards from '../components/StatsCards'
import CodeForm from '../components/CodeForm'
import PipelineProgress from '../components/PipelineProgress'
import ResultsPanel from '../components/ResultsPanel'
import TaskHistory from '../components/TaskHistory'

export default function Dashboard({ user, onLogout }) {
  const [mode, setMode] = useState('analyze') // 'analyze' | 'fix'
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [tasks, setTasks] = useState([])
  const [health, setHealth] = useState(null)

  useEffect(() => {
    healthCheck().then(setHealth).catch(() => {})
    loadTasks()
  }, [])

  const loadTasks = async () => {
    try {
      const data = await getUserTasks(user)
      setTasks(data.tasks || [])
    } catch (err) {
      // silently fail on task load
    }
  }

  const handleSubmit = async ({ code, language }) => {
    setRunning(true)
    setError('')
    setResult(null)

    try {
      const fn = mode === 'fix' ? fixBug : analyzeBug
      const data = await fn(code, language)
      setResult(data)
      loadTasks() // refresh task history
    } catch (err) {
      setError(err.message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">🐛 BugFix</div>
        <nav className="sidebar-nav">
          <Link to="/" className="active">📋 代码分析</Link>
          <button onClick={loadTasks}>🔄 刷新记录</button>
        </nav>
        <div className="sidebar-user">
          <div style={{ fontWeight: 500, color: '#c9d1d9' }}>{user}</div>
          <button onClick={onLogout} style={{
            background: 'none', border: 'none', color: 'var(--text-dim)',
            cursor: 'pointer', fontSize: '0.85em', marginTop: 4, padding: 0
          }}>
            退出登录
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="main-content">
        <StatsCards health={health} tasks={tasks} />

        {/* Mode switcher */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
          <button
            className={`btn ${mode === 'analyze' ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => { setMode('analyze'); setResult(null); setError('') }}
            disabled={running}
          >
            🔍 Bug 分析
          </button>
          <button
            className={`btn ${mode === 'fix' ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => { setMode('fix'); setResult(null); setError('') }}
            disabled={running}
          >
            🔧 完整修复
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
          {/* Left: Input */}
          <div className="card">
            <div className="card-title">📥 提交代码</div>
            <CodeForm onSubmit={handleSubmit} running={running} />
            {error && <div className="error-msg" style={{ marginTop: 12 }}>{error}</div>}
          </div>

          {/* Right: Results */}
          <div className="card">
            <div className="card-title">
              📤 {mode === 'fix' ? '修复结果' : '分析结果'}
            </div>
            {running ? (
              <div className="loading-overlay">
                <span className="spinner" />
                <span>
                  {mode === 'fix'
                    ? 'Multi-Agent 协同修复中 (约60-120秒)...'
                    : 'Agent 分析中 (约20-40秒)...'}
                </span>
              </div>
            ) : result ? (
              <ResultsPanel result={result} mode={mode} />
            ) : (
              <div className="empty-state">
                <div className="empty-state-icon">📝</div>
                <p>提交代码开始{mode === 'fix' ? '修复' : '分析'}</p>
              </div>
            )}
          </div>
        </div>

        {/* Pipeline */}
        {running && <PipelineProgress mode={mode} />}
        {result?.steps && result.steps.length > 0 && (
          <div className="card" style={{ marginTop: 20 }}>
            <div className="card-title">📊 执行步骤</div>
            <PipelineProgress mode={mode} steps={result.steps} completed />
          </div>
        )}

        {/* Task History */}
        <div className="card" style={{ marginTop: 20 }}>
          <div className="card-title">📋 任务记录</div>
          <TaskHistory tasks={tasks} />
        </div>
      </main>
    </div>
  )
}
