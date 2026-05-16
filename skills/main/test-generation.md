---
name: test-generation
description: 单元测试生成 - 编写验证Bug修复的测试用例
version: 2.0.0
scope: main
tags: [testing, validation, junit, java]
dependencies: [fix-patterns]
created_at: 2026-05-16T00:00:00
---

# Test Generation Skill

## When to use
当需要生成测试用例验证Bug修复时使用。

## 测试用例模板

### Template 1: Null Input Test
```java
@Test(expected = IllegalArgumentException.class)
public void testNullInput() {
    MyClass obj = new MyClass();
    obj.process(null); // 应抛异常
}
```

### Template 2: Boundary Test
```java
@Test
public void testBoundary() {
    MyClass obj = new MyClass();

    // 正常边界
    assertEquals(0, obj.getIndex(0));
    assertEquals(size-1, obj.getIndex(size-1));

    // 越界边界
    assertThrows(IndexOutOfBoundsException.class,
        () -> obj.getIndex(-1));
    assertThrows(IndexOutOfBoundsException.class,
        () -> obj.getIndex(size));
}
```

### Template 3: Negative Input Test
```java
@Test
public void testNegativeInput() {
    MyClass obj = new MyClass();

    // 负数处理
    assertEquals(-1, obj.calculate(-5));

    // 负数边界
    assertThrows(IllegalArgumentException.class,
        () -> obj.calculate(-999));
}
```

### Template 4: Edge Case Test
```java
@Test
public void testEdgeCases() {
    MyClass obj = new MyClass();

    // 空输入
    assertEquals("", obj.process(""));

    // 最大值
    assertEquals(Long.MAX_VALUE, obj.processMax(Long.MAX_VALUE));

    // 最小值
    assertEquals(Long.MIN_VALUE, obj.processMin(Long.MIN_VALUE));
}
```

## Defects4J测试参考

### Math Bug #1 测试用例
```java
@Test
public void testFractionConversion() {
    // Bug场景：epsilon=0时应该抛异常，不应该break
    double value = 0.123456789;

    assertThrows(FractionConversionException.class,
        () -> new BigFraction(value, 0, 100));
}
```

## 测试覆盖率目标

| Bug类型 | 需覆盖场景 |
|---------|------------|
| NPE | null输入、链式调用null |
| 边界 | 正常值、边界值、越界值 |
| 逻辑 | 正确路径、错误路径 |
| 性能 | 正常负载、极端负载 |

## 测试验证流程

1. **运行原Bug测试**
   - 确认原测试失败
   - 记录失败原因

2. **运行修复后测试**
   - 确认测试通过
   - 检查无新引入Bug

3. **回归测试**
   - 运行全量测试
   - 确认无副作用