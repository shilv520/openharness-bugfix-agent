export default function StatsCards({ health, tasks }) {
  const completedTasks = tasks.filter(t =>
    t.status === 'completed' || t.status === 'fixed'
  ).length

  return (
    <div className="stats-grid">
      <div className="stat-card">
        <div className="stat-value">{health?.version || '2.0'}</div>
        <div className="stat-label">API 版本</div>
      </div>
      <div className="stat-card">
        <div className="stat-value">4</div>
        <div className="stat-label">协作 Agents</div>
      </div>
      <div className="stat-card">
        <div className="stat-value">{tasks.length}</div>
        <div className="stat-label">总任务数</div>
      </div>
      <div className="stat-card">
        <div className="stat-value">{completedTasks}</div>
        <div className="stat-label">已完成</div>
      </div>
    </div>
  )
}
