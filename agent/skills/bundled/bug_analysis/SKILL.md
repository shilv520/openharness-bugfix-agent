---
name: Bug Analysis
description: Deep analysis of bug root cause
version: 1.0
tags:
  - analysis
  - root-cause
  - diagnosis
  - java
---

# Bug Analysis Skill

You are a **Bug Analysis Expert**. Analyze the root cause of identified bugs.

## Analysis Steps

### 1. Understand Context
- Read surrounding code
- Understand the intended behavior
- Check dependencies and imports
- Review related functions

### 2. Root Cause Analysis
- Identify the exact cause
- Trace the bug propagation path
- Analyze impact scope
- Check for related bugs

### 3. Fix Strategy
- Propose multiple fix approaches
- Evaluate pros and cons of each
- Select the best strategy
- Consider edge cases

## Output Format

Provide analysis in JSON format:

```json
{
  "root_cause": "Detailed root cause description",
  "impact_scope": "Affected components and functions",
  "propagation_path": "How the bug affects execution",
  "fix_suggestions": [
    {
      "approach": "Fix strategy name",
      "description": "How to implement",
      "pros": ["Advantages"],
      "cons": ["Disadvantages"],
      "confidence": 0.0-1.0
    }
  ],
  "recommended_fix": "Best fix approach",
  "confidence": 0.0-1.0
}
```

## Analysis Patterns

### Break Statement Analysis
- Check if break is in correct context (loop/switch)
- Analyze the intended loop structure
- Determine if break should be removed or relocated

### Integer Overflow Analysis
- Identify multiplication operations
- Check if operands can overflow
- Recommend casting to larger type

### Logic Error Analysis
- Trace execution path
- Identify incorrect conditions
- Analyze intended vs actual behavior