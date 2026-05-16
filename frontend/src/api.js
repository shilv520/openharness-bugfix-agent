const BASE = '/api'

function token() {
  return localStorage.getItem('token') || ''
}

function authHeaders() {
  const t = token()
  return t ? { Authorization: `Bearer ${t}` } : {}
}

async function request(method, path, body = null, timeout = 180000) {
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), timeout)

  const opts = {
    method,
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    signal: controller.signal,
  }
  if (body) opts.body = JSON.stringify(body)

  const resp = await fetch(`${BASE}${path}`, opts)
  clearTimeout(id)

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || `HTTP ${resp.status}`)
  }
  return resp.json()
}

// Auth
export const register = (username, password, email) =>
  request('POST', '/register', { username, password, email })

export const login = (username, password) =>
  request('POST', '/login', { username, password })

// Bug operations
export const analyzeBug = (code, language, description) =>
  request('POST', '/bug/analyze', { code, language, description }, 180000)

export const fixBug = (code, language, description) =>
  request('POST', '/bug/fix', { code, language, description }, 300000)

export const submitBugAsync = (code, language, description, username = 'anonymous') =>
  request('POST', `/bug/submit?username=${encodeURIComponent(username)}`, { code, language, description })

// Task queries
export const getTask = (taskId) =>
  request('GET', `/bug/task/${taskId}`)

export const getUserTasks = (username) =>
  request('GET', `/bug/tasks/${encodeURIComponent(username)}`)

// Health
export const healthCheck = () =>
  request('GET', '/health', null, 10000)
