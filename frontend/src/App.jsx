import { Routes, Route, Navigate } from 'react-router-dom'
import { useState } from 'react'
import LoginPage from './pages/LoginPage'
import Dashboard from './pages/Dashboard'
import TaskDetail from './pages/TaskDetail'

export default function App() {
  const [user, setUser] = useState(() => {
    const u = localStorage.getItem('username')
    return u || null
  })

  const handleLogin = (username, tok) => {
    localStorage.setItem('username', username)
    localStorage.setItem('token', tok)
    setUser(username)
  }

  const handleLogout = () => {
    localStorage.removeItem('username')
    localStorage.removeItem('token')
    setUser(null)
  }

  if (!user) {
    return <LoginPage onLogin={handleLogin} />
  }

  return (
    <Routes>
      <Route path="/" element={<Dashboard user={user} onLogout={handleLogout} />} />
      <Route path="/task/:taskId" element={<TaskDetail user={user} onLogout={handleLogout} />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
