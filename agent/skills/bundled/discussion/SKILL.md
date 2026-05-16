---
name: Agent Discussion
description: Protocol for multi-agent collaboration and consensus
version: 1.0
tags:
  - collaboration
  - discussion
  - consensus
  - multi-agent
---

# Agent Discussion Skill

Protocol for agents to discuss findings and reach consensus.

## Discussion Format

### 1. Initial Position (Agent A)
- Present initial analysis/findings
- Include reasoning and evidence
- State confidence level

### 2. Response (Agent B)
- Review Agent A's position
- Agree or provide alternative view
- Include supporting reasoning

### 3. Consensus
- Both agents agree on final position
- Record agreed conclusion
- Document any remaining concerns

## Discussion Protocol

### Reviewer <-> Analyzer Discussion

**Step 1: Reviewer Presents**
```json
{
  "agent": "Reviewer",
  "position": {
    "bug_found": true|false,
    "location": "bug location",
    "type": "bug type",
    "reasoning": "Why this is a bug"
  },
  "confidence": 0.0-1.0
}
```

**Step 2: Analyzer Responds**
```json
{
  "agent": "Analyzer",
  "response": {
    "agreed": true|false,
    "confirmed_bug": {
      "location": "confirmed location",
      "type": "confirmed type"
    },
    "additional_analysis": "Extra insights"
  },
  "confidence": 0.0-1.0
}
```

**Step 3: Consensus**
```json
{
  "consensus": {
    "bug_location": "agreed location",
    "bug_type": "agreed type",
    "confidence": "combined confidence"
  }
}
```

### Analyzer <-> Fixer Discussion

**Step 1: Analyzer Presents Fix Strategy**
```json
{
  "agent": "Analyzer",
  "fix_suggestion": {
    "approach": "fix approach name",
    "description": "how to fix",
    "rationale": "why this approach"
  }
}
```

**Step 2: Fixer Responds**
```json
{
  "agent": "Fixer",
  "response": {
    "agreed": true|false,
    "implementation": "how will implement",
    "concerns": ["any concerns"]
  }
}
```

### Fixer <-> Validator Feedback Loop

**Iteration 1: Validator Review**
```json
{
  "agent": "Validator",
  "validation": {
    "passed": true|false,
    "issues": ["found issues"],
    "feedback": "what to improve"
  }
}
```

**Iteration 2: Fixer Adjustment**
```json
{
  "agent": "Fixer",
  "adjustment": {
    "changes": "what changed",
    "rationale": "why changed"
  }
}
```

**Continue until validation passes or max iterations**

## Best Practices

1. Always include reasoning with positions
2. State confidence explicitly
3. Provide evidence when possible
4. Be open to alternative views
5. Document consensus clearly