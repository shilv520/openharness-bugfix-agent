import { useState } from 'react'

const DEFAULT_CODE = `public class BigFraction {
    private long overflow = 100000;
    public BigFraction(double value) {
        long p2 = 0, q2 = 1;
        if ((p2 > overflow) || (q2 > overflow)) {
            if (Math.abs(q2) < overflow) {
                break;  // BUG: break not in loop!
            }
            throw new RuntimeException();
        }
    }
}`

export default function CodeForm({ onSubmit, running }) {
  const [code, setCode] = useState(DEFAULT_CODE)
  const [language, setLanguage] = useState('java')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!code.trim()) return
    onSubmit({ code, language })
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-group">
        <label className="form-label">编程语言</label>
        <select
          className="form-select"
          value={language}
          onChange={e => setLanguage(e.target.value)}
          disabled={running}
        >
          <option value="java">Java</option>
          <option value="python">Python</option>
          <option value="javascript">JavaScript</option>
          <option value="typescript">TypeScript</option>
          <option value="go">Go</option>
          <option value="rust">Rust</option>
        </select>
      </div>

      <div className="form-group">
        <label className="form-label">
          源代码
          <span style={{ color: 'var(--text-dim)', fontWeight: 400, marginLeft: 8 }}>
            ({code.length} 字符)
          </span>
        </label>
        <textarea
          className="form-textarea"
          value={code}
          onChange={e => setCode(e.target.value)}
          placeholder="粘贴有Bug的代码..."
          style={{ height: 260 }}
          disabled={running}
        />
      </div>

      <button
        type="submit"
        className="btn btn-primary"
        disabled={running || !code.trim()}
        style={{ width: '100%', justifyContent: 'center' }}
      >
        {running ? (
          <><span className="spinner" /> 分析中...</>
        ) : (
          '🔍 开始分析'
        )}
      </button>
    </form>
  )
}
