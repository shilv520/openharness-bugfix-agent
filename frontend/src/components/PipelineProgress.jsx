const STEPS_MAP = {
  review: { label: 'Reviewer 审查', icon: '🔍' },
  discuss_review_analyzer: { label: 'Reviewer↔Analyzer 讨论', icon: '💬' },
  analyze: { label: 'Analyzer 根因分析', icon: '🔬' },
  discuss_analyzer_fixer: { label: 'Analyzer↔Fixer 讨论', icon: '💬' },
  fix: { label: 'Fixer 生成补丁', icon: '🔧' },
  validate: { label: 'Validator 验证', icon: '✅' },
  no_bugs_found: { label: '无Bug发现', icon: '⏭️' },
}

export default function PipelineProgress({ mode, steps = [], completed = false }) {
  // If running (no completed steps yet), show all steps with active first
  if (!completed) {
    const pipeline = mode === 'fix'
      ? ['review', 'discuss_review_analyzer', 'analyze', 'discuss_analyzer_fixer', 'fix', 'validate']
      : ['review', 'discuss_review_analyzer', 'analyze']

    return (
      <div className="card">
        <div className="card-title">⚙️ 执行中...</div>
        <div className="pipeline">
          {pipeline.map((key, i) => {
            const step = STEPS_MAP[key] || { label: key, icon: '⏺️' }
            return (
              <span key={key}>
                {i > 0 && <span className="pipeline-arrow">→</span>}
                <span className={`pipeline-step ${i === 0 ? 'active' : 'waiting'}`}>
                  {step.icon} {step.label}
                </span>
              </span>
            )
          })}
        </div>
      </div>
    )
  }

  // Show completed steps
  return (
    <div className="pipeline">
      {steps.map((s, i) => {
        const key = s.step || s
        const info = STEPS_MAP[key] || { label: key, icon: '⏺️' }
        const passed = s.passed ?? s.agreed ?? s.patch_generated ?? s.result === 'success'
        const state = passed ? 'done' : 'fail'

        return (
          <span key={i}>
            {i > 0 && <span className="pipeline-arrow">→</span>}
            <span className={`pipeline-step ${state}`}>
              {info.icon} {info.label}
            </span>
          </span>
        )
      })}
    </div>
  )
}
