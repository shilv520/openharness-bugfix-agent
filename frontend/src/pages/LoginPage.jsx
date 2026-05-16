import { useState } from 'react'
import { login, register } from '../api'

export default function LoginPage({ onLogin }) {
  const [tab, setTab] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (!username.trim() || !password.trim()) {
      setError('用户名和密码不能为空')
      return
    }
    setLoading(true)
    try {
      if (tab === 'register') {
        await register(username, password, email || undefined)
        setTab('login')
        setError('注册成功，请登录')
      } else {
        const data = await login(username, password)
        if (data.token) {
          onLogin(username, data.token)
        }
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-container">
      <div className="login-card">
        <h1 className="login-title">🐛 Bug Fix Agent</h1>
        <p className="login-subtitle">Multi-Agent 智能代码修复系统</p>

        <div className="login-tabs">
          <button
            className={`login-tab ${tab === 'login' ? 'active' : ''}`}
            onClick={() => { setTab('login'); setError('') }}
          >
            登录
          </button>
          <button
            className={`login-tab ${tab === 'register' ? 'active' : ''}`}
            onClick={() => { setTab('register'); setError('') }}
          >
            注册
          </button>
        </div>

        {error && <div className={`error-msg ${error.includes('成功') ? '' : ''}`} style={error.includes('成功') ? { background: '#0d3320', color: '#3fb950' } : {}}>
          {error}
        </div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">用户名</label>
            <input
              className="form-input"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="输入用户名"
              autoFocus
            />
          </div>

          {tab === 'register' && (
            <div className="form-group">
              <label className="form-label">邮箱（可选）</label>
              <input
                className="form-input"
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="输入邮箱"
              />
            </div>
          )}

          <div className="form-group">
            <label className="form-label">密码</label>
            <input
              className="form-input"
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="输入密码"
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary btn-lg"
            style={{ width: '100%', justifyContent: 'center', marginTop: 8 }}
            disabled={loading}
          >
            {loading && <span className="spinner" />}
            {tab === 'login' ? '登录' : '注册'}
          </button>
        </form>
      </div>
    </div>
  )
}
