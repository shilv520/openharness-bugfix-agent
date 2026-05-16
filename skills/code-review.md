---
name: code-review
description: Java代码审查 - 识别潜在Bug和质量问题
---

# Code Review Skill

## When to use
当需要审查Java代码，识别潜在Bug时使用。

## Common Bug Types

| Bug类型 | 特征 | 代码模式 |
|---------|------|----------|
| 空指针异常 | 未检查null | `obj.method()` 前无null检查 |
| 逻辑错误 | 条件判断不当 | `if (x > 0)` 应为 `if (x >= 0)` |
| 边界条件 | 数组越界、负数处理 | `array[i]` 未检查i范围 |
| 资源泄漏 | 未关闭流/连接 | `InputStream` 未close |
| 并发问题 | 死锁、竞态条件 | 缺少synchronized |

## Review Checklist

1. **空指针检查**
   - 方法参数是否检查null
   - 返回值是否可能为null
   - 链式调用是否安全

2. **边界条件**
   - 数组访问是否检查索引
   - 循环边界是否正确 (0 vs 1)
   - 负数输入是否处理

3. **异常处理**
   - 是否捕获必要异常
   - 异常信息是否完整
   - 是否有fallback逻辑

4. **资源管理**
   - 文件/流是否关闭
   - 连接是否释放
   - try-with-resources使用

## Bug Pattern Examples

### 1. Null Pointer Pattern
```java
// Bug: 未检查null
public String getName(User user) {
    return user.getName(); // user可能为null
}

// Fix: 添加null检查
public String getName(User user) {
    if (user == null) return null;
    return user.getName();
}
```

### 2. Logic Error Pattern
```java
// Bug: 条件判断错误
if (epsilon == 0.0 && FastMath.abs(q1) < maxDenominator) {
    break; // 应该抛异常而不是break
}

// Fix: 删除错误逻辑
throw new FractionConversionException(value, p2, q2);
```

### 3. Boundary Condition Pattern
```java
// Bug: 数组越界风险
for (int i = 0; i <= array.length; i++) { // 应为 < 而非 <=
    process(array[i]);
}

// Fix: 正确边界
for (int i = 0; i < array.length; i++) {
    process(array[i]);
}
```