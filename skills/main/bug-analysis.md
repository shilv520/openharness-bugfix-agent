---
name: bug-analysis
description: Bug根因分析 - 定位Bug类型和具体位置
version: 2.0.0
scope: main
tags: [analysis, root-cause, diagnosis, java]
dependencies: [code-review]
created_at: 2026-05-16T00:00:00
---

# Bug Analysis Skill

## When to use
当需要深入分析Bug根因，确定Bug类型和位置时使用。

## Bug分类体系

### Level 1: Bug大类
- **Syntax Errors**: 编译错误（类型不匹配、语法错误）
- **Runtime Errors**: 运行时异常（NPE、越界）
- **Logic Errors**: 逻辑错误（算法错误、条件错误）
- **Performance Issues**: 性能问题（死循环、内存泄漏）

### Level 2: 具体类型

| 大类 | 子类型 | 示例 |
|------|--------|------|
| Runtime | NullPointerException | 未检查null调用方法 |
| Runtime | ArrayIndexOutOfBounds | 索引超出数组范围 |
| Runtime | ClassCastException | 类型转换错误 |
| Logic | Off-by-one | 循环边界错误 |
| Logic | Condition Error | if/else条件颠倒 |
| Logic | Missing Branch | 缺少else分支 |
| Performance | Infinite Loop | 循环条件永不终止 |

## 根因分析方法

### Step 1: 定位Bug位置
1. 查看异常堆栈，定位出错行号
2. 分析出错行的代码逻辑
3. 检查变量状态（是否为null、超出范围）

### Step 2: 分析Bug类型
1. 根据异常类型判断大类
2. 根据代码模式判断子类型
3. 确认是否为组合问题（多种Bug叠加）

### Step 3: 确定根因
1. 分析代码意图（原本应该做什么）
2. 对比实际行为（实际做了什么）
3. 找出意图与行为的差异点

## 分析模板

```markdown
## Bug分析报告

### Bug位置
- 文件: {filename}
- 行号: {line_number}
- 函数: {method_name}

### Bug类型
- 大类: {category}
- 子类型: {sub_type}

### 根因分析
- 代码意图: {intended_behavior}
- 实际行为: {actual_behavior}
- 差异原因: {root_cause}

### 影响范围
- 直接影响: {direct_impact}
- 潜在影响: {potential_impact}

### 修复建议
- 修复类型: {fix_type}
- 修复优先级: {priority}
```

## 常见根因模式

### NPE根因模式
```
根因: 调用链中某对象可能为null，但未检查
位置: obj.method() 或 obj.field
修复: 添加null检查或使用Optional
```

### 边界错误根因模式
```
根因: 循环/数组边界判断错误（< vs <=）
位置: for循环条件或数组访问
修复: 修正边界条件
```

### 逻辑错误根因模式
```
根因: 条件判断与业务逻辑不符
位置: if/else/switch语句
修复: 调整条件逻辑或分支顺序
```