import { Link } from 'react-router-dom'

const STATUS_MAP = {
  pending: { label: '等待中', cls: 'badge-pending' },
  analyzed: { label: '已分析', cls: 'badge-info' },
  completed: { label: '已完成', cls: 'badge-success' },
  fixed: { label: '已修复', cls: 'badge-success' },
  failed: { label: '失败', cls: 'badge-error' },
}

export default function TaskHistory({ tasks }) {
  if (!tasks || tasks.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📭</div>
        <p>暂无任务记录</p>
        <p style={{ fontSize: '0.85em', marginTop: 4 }}>提交代码分析后将显示在这里</p>
      </div>
    )
  }

  const sorted = [...tasks].reverse()

  return (
    <div className="table-container">
      <table>
        <thead>
          <tr>
            <th>任务 ID</th>
            <th>语言</th>
            <th>Bug类型</th>
            <th>根因</th>
            <th>状态</th>
            <th>时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(task => {
            const st = STATUS_MAP[task.status] || { label: task.status || 'unknown', cls: 'badge-info' }
            return (
              <tr key={task.task_id || Math.random()}>
                <td style={{ fontFamily: 'monospace', fontSize: '0.8em' }}>
                  {(task.task_id || '?').replace('task:', '')}
                </td>
                <td>{task.language || '-'}</td>
                <td>
                  {task.bug_type ? (
                    <span className="badge badge-info">{task.bug_type}</span>
                  ) : '-'}
                </td>
                <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {task.root_cause || task.error || '-'}
                </td>
                <td><span className={`badge ${st.cls}`}>{st.label}</span></td>
                <td style={{ fontSize: '0.8em', color: 'var(--text-dim)' }}>
                  {task.created_at ? new Date(task.created_at).toLocaleString('zh-CN') : '-'}
                </td>
                <td>
                  <Link
                    to={`/task/${(task.task_id || '').replace('task:', '')}`}
                    className="btn btn-outline btn-sm"
                  >
                    详情
                  </Link>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
