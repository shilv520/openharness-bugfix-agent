export default function ResultsPanel({ result, mode }) {
  const { success, review, discussion, analysis, patch, fixed_code, test_passed, total_steps, steps, error } = result

  if (error && !analysis?.root_cause) {
    return (
      <div className="error-msg">
        <strong>执行失败:</strong> {error}
      </div>
    )
  }

  return (
    <div>
      {/* Status */}
      <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 10 }}>
        <span className={`badge ${success ? 'badge-success' : 'badge-error'}`}>
          {success ? '✓ 成功' : '✗ 失败'}
        </span>
        {total_steps > 0 && (
          <span className="badge badge-info">{total_steps} 步骤</span>
        )}
      </div>

      {/* Review summary */}
      {review && (
        <div className="result-field">
          <div className="result-field-label">代码审查</div>
          <div className="result-field-value">
            发现 <strong>{review.bugs_found || 0}</strong> 个潜在Bug
            {review.code_quality && (
              <span style={{ marginLeft: 8, color: 'var(--text-dim)' }}>
                | 代码质量: {review.code_quality}
              </span>
            )}
          </div>
          {review.candidates && review.candidates.length > 0 && (
            <div style={{ marginTop: 8 }}>
              {review.candidates.map((bug, i) => (
                <div key={i} style={{
                  background: 'var(--bg)',
                  padding: '8px 12px',
                  borderRadius: 'var(--radius)',
                  marginBottom: 6,
                  fontSize: '0.85em'
                }}>
                  <span className={`badge ${bug.severity === 'high' ? 'badge-error' : 'badge-pending'}`} style={{ marginRight: 8 }}>
                    {bug.type || 'unknown'}
                  </span>
                  {bug.description || bug.location}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Analysis */}
      {analysis && (
        <div className="result-grid" style={{ marginTop: 16 }}>
          <div className="result-field">
            <div className="result-field-label">Bug 位置</div>
            <div className="result-field-value">{analysis.bug_location || 'N/A'}</div>
          </div>
          <div className="result-field">
            <div className="result-field-label">Bug 类型</div>
            <div className="result-field-value">{analysis.bug_type || 'N/A'}</div>
          </div>
          <div className="result-field">
            <div className="result-field-label">置信度</div>
            <div className="result-field-value">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{
                  flex: 1, height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${(analysis.confidence || 0) * 100}%`,
                    height: '100%',
                    background: 'var(--primary)',
                    borderRadius: 3,
                    transition: 'width 0.5s'
                  }} />
                </div>
                <span>{((analysis.confidence || 0) * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
          <div className="result-field">
            <div className="result-field-label">测试结果</div>
            <div className="result-field-value">
              <span className={`badge ${test_passed ? 'badge-success' : 'badge-error'}`}>
                {test_passed ? '通过' : '未通过'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Root cause */}
      {analysis?.root_cause && (
        <div className="result-field" style={{ marginTop: 16 }}>
          <div className="result-field-label">根因分析</div>
          <div className="code-block" style={{ maxHeight: 120, fontSize: '0.85em' }}>
            {analysis.root_cause}
          </div>
        </div>
      )}

      {/* Fix suggestion */}
      {analysis?.fix_suggestion && (
        <div className="result-field" style={{ marginTop: 12 }}>
          <div className="result-field-label">修复建议</div>
          <div className="result-field-value" style={{ fontSize: '0.9em' }}>
            {analysis.fix_suggestion}
          </div>
        </div>
      )}

      {/* Patch (full fix mode) */}
      {patch && (
        <div className="result-field" style={{ marginTop: 16 }}>
          <div className="result-field-label">修复补丁</div>
          <div className="code-block" style={{
            maxHeight: 200,
            borderLeft: '3px solid var(--success)',
          }}>
            {patch}
          </div>
        </div>
      )}

      {/* Fixed code */}
      {fixed_code && (
        <div className="result-field" style={{ marginTop: 12 }}>
          <div className="result-field-label">修复后代码</div>
          <div className="code-block" style={{
            maxHeight: 200,
            borderLeft: '3px solid var(--primary)',
          }}>
            {fixed_code}
          </div>
        </div>
      )}

      {/* Discussion consensus */}
      {discussion?.consensus && discussion.consensus !== 'None' && (
        <div className="result-field" style={{ marginTop: 12 }}>
          <div className="result-field-label">
            Agent 讨论共识
            <span style={{ marginLeft: 8, fontWeight: 400 }}>
              ({discussion.agreed ? '达成一致' : '未达成一致'})
            </span>
          </div>
          <div className="result-field-value" style={{ fontSize: '0.85em', color: 'var(--text-dim)' }}>
            {discussion.consensus}
          </div>
        </div>
      )}
    </div>
  )
}
