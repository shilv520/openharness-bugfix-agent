---
name: fix-patterns
description: 代码修复模式库 - 提供常见Bug修复模板和补丁示例
---

# Fix Patterns Skill

## When to use
当需要生成修复补丁时，参考此Skill中的修复模式。

## 修复模式库

### Pattern 1: Null Check Addition
**适用Bug**: NullPointerException

```diff
--- original
+++ fixed
@@ -X,Y +X,Y
-     return obj.method();
+     if (obj == null) {
+         throw new IllegalArgumentException("obj cannot be null");
+     }
+     return obj.method();
```

### Pattern 2: Boundary Fix (Off-by-one)
**适用Bug**: ArrayIndexOutOfBoundsException, 循环边界错误

```diff
--- original
+++ fixed
@@ -X,Y +X,Y
-     for (int i = 0; i <= array.length; i++) {
+     for (int i = 0; i < array.length; i++) {
```

### Pattern 3: Condition Logic Fix
**适用Bug**: 逻辑错误，条件判断不当

```diff
--- original
+++ fixed
@@ -X,Y +X,Y
-     if (epsilon == 0.0 && abs(q1) < max) {
-         break;
-     }
-     throw new Exception();
+     throw new Exception(); // 删除错误的break逻辑
```

### Pattern 4: Missing Branch Addition
**适用Bug**: 缺少else分支处理

```diff
--- original
+++ fixed
@@ -X,Y +X,Y
      if (condition) {
          handleA();
      }
+     else {
+         handleB(); // 添加缺失分支
+     }
```

### Pattern 5: Exception Handling Fix
**适用Bug**: 异常处理不当

```diff
--- original
+++ fixed
@@ -X,Y +X,Y
      try {
          riskyOperation();
-     } catch (Exception e) {} // 空catch
+     } catch (Exception e) {
+         logger.error("Operation failed", e);
+         throw new RuntimeException(e);
+     }
```

### Pattern 6: Resource Leak Fix
**适用Bug**: 资源未关闭

```diff
--- original
+++ fixed
@@ -X,Y +X,Y
-     InputStream in = new FileInputStream(file);
-     process(in);
-     in.close(); // 可能异常导致不执行
+     try (InputStream in = new FileInputStream(file)) {
+         process(in);
+     } // try-with-resources自动关闭
```

## 补丁生成模板

```python
def generate_patch(bug_type, bug_location, original_code):
    """生成修复补丁"""

    patterns = {
        "null_pointer": NULL_CHECK_PATTERN,
        "off_by_one": BOUNDARY_FIX_PATTERN,
        "logic_error": CONDITION_FIX_PATTERN,
        "missing_branch": BRANCH_ADDITION_PATTERN,
        "exception_handling": EXCEPTION_FIX_PATTERN,
        "resource_leak": RESOURCE_FIX_PATTERN
    }

    pattern = patterns.get(bug_type)
    if not pattern:
        return None

    # 根据bug_location填充补丁
    patch = apply_pattern(pattern, bug_location, original_code)
    return patch
```

## 补丁验证清单

1. **语法正确性**
   - 补丁后代码可编译
   - 类型匹配正确

2. **语义正确性**
   - 修复了原Bug
   - 未改变正确逻辑

3. **无副作用**
   - 未引入新Bug
   - 其他测试仍通过

4. **代码质量**
   - 符合代码规范
   - 注释清晰

## Defects4J补丁示例

### Math Bug #1 修复补丁
```diff
--- a/src/main/java/org/apache/commons/math3/fraction/BigFraction.java
+++ b/src/main/java/org/apache/commons/math3/fraction/BigFraction.java
@@ -303,9 +303,6 @@
             if ((p2 > overflow) || (q2 > overflow)) {
-                if (epsilon == 0.0 && FastMath.abs(q1) < maxDenominator) {
-                    break;
-                }
                 throw new FractionConversionException(value, p2, q2);
             }
```

**修复类型**: Logic Error - 删除错误的break逻辑
**Bug ID**: MATH-996