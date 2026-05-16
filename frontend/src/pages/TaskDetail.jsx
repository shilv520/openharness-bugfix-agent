import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getTask } from '../api'
import ResultsPanel from '../components/ResultsPanel'
import PipelineProgress from '../components/PipelineProgress'

export default function TaskDetail({ user, onLogout }) {
  const { taskId } = useParams()
  const [task, setTask] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => { loadTask() }, [taskId])

  const loadTask = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await getTask(taskId)
      setTask(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-logo">🐛 BugFix</div>
        <nav className="sidebar-nav">
          <Link to="/">📋 代码分析</Link>
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

      <main className="main-content">
        <div style={{ marginBottom: 20 }}>
          <Link to="/" style={{ color: 'var(--text-dim)', fontSize: '0.9em' }}>
            ← 返回仪表盘
          </Link>
        </div>

        {loading ? (
          <div className="loading-overlay">
            <span className="spinner" />
            <span>加载任务详情...</span>
          </div>
        ) : error ? (
          <div className="error-msg">{error}</div>
        ) : task ? (
          <>
            <div className="card">
              <div className="card-title">
                📋 任务 {task.task_id || taskId}
              </div>

              <div className="result-grid">
                <div className="result-field">
                  <div className="result-field-label">状态</div>
                  <div className="result-field-value">
                    <span className={`badge ${task.status === 'fixed' || task.status === 'completed' ? 'badge-success' : task.status === 'failed' ? 'badge-error' : 'badge-info'}`}>
                      {task.status || 'unknown'}
                    </span>
                  </div>
                </div>
                <div className="result-field">
                  <div className="result-field-label">语言</div>
                  <div className="result-field-value">{task.language || '-'}</div>
                </div>
                <div className="result-field">
                  <div className="result-field-label">Bug 类型</div>
                  <div className="result-field-value">{task.bug_type || '-'}</div>
                </div>
                <div className="result-field">
                  <div className="result-field-label">置信度</div>
                  <div className="result-field-value">
                    {task.confidence ? `${(parseFloat(task.confidence) * 100).toFixed(0)}%` : '-'}
                  </div>
                </div>
              </div>

              {task.root_cause && (
                <div className="result-field" style={{ marginTop: 16 }}>
                  <div className="result-field-label">根因分析</div>
                  <div className="code-block" style={{ maxHeight: 120 }}>{task.root_cause}</div>
                </div>
              )}

              {task.fix_suggestion && (
                <div className="result-field" style={{ marginTop: 12 }}>
                  <div className="result-field-label">修复建议</div>
                  <div className="result-field-value">{task.fix_suggestion}</div>
                </div>
              )}
            </div>

            {/* Pipeline Steps */}
            {task.steps && (
              <div className="card">
                <div className="card-title">📊 执行步骤 ({task.total_steps || task.steps.length})</div>
                <PipelineProgress steps={task.steps} completed />
              </div>
            )}

            {/* Patch & Fixed Code */}
            {(task.patch || task.fixed_code) && (
              <div className="card">
                <div className="card-title">🔧 修复详情</div>
                {task.patch && (
                  <div className="result-field">
                    <div className="result-field-label">补丁</div>
                    <div className="code-block" style={{ borderLeft: '3px solid var(--success)', maxHeight: 300 }}>
                      {task.patch}
                    </div>
                  </div>
                )}
                {task.fixed_code && (
                  <div className="result-field" style={{ marginTop: 16 }}>
                    <div className="result-field-label">修复后代码</div>
                    <div className="code-block" style={{ borderLeft: '3px solid var(--primary)', maxHeight: 300 }}>
                      {task.fixed_code}
                    </div>
                  </div>
                )}
                {task.test_passed !== undefined && (
                  <div className="result-field" style={{ marginTop: 12 }}>
                    <div className="result-field-label">测试验证</div>
                    <span className={`badge ${task.test_passed === 'True' || task.test_passed === true ? 'badge-success' : 'badge-error'}`}>
                      {task.test_passed === 'True' || task.test_passed === true ? '✓ 通过' : '✗ 未通过'}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Original Code */}
            {task.code && (
              <div className="card">
                <div className="card-title">📝 原始代码</div>
                <div className="code-block">{task.code}</div>
              </div>
            )}

            {/* Metadata */}
            <div className="card">
              <div className="card-title">📎 元数据</div>
              <div className="result-grid">
                <div className="result-field">
                  <div className="result-field-label">用户</div>
                  <div className="result-field-value">{task.username || '-'}</div>
                </div>
                <div className="result-field">
                  <div className="result-field-label">创建时间</div>
                  <div className="result-field-value">
                    {task.created_at ? new Date(task.created_at).toLocaleString('zh-CN') : '-'}
                  </div>
                </div>
                <div className="result-field">
                  <div className="result-field-label">总步骤数</div>
                  <div className="result-field-value">{task.total_steps || '-'}</div>
                </div>
                {task.error && (
                  <div className="result-field">
                    <div className="result-field-label">错误</div>
                    <div className="result-field-value" style={{ color: 'var(--danger)' }}>{task.error}</div>
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">❓</div>
            <p>任务不存在</p>
          </div>
        )}
      </main>
    </div>
  )
}
