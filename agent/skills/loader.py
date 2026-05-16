"""
Skill 动态加载机制
==================

实现 OpenHarness 的 Skills 动态加载模式
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SkillDefinition:
    """Skill 定义"""
    name: str
    description: str
    content: str
    source: str  # bundled | user | plugin
    path: Optional[str] = None
    version: str = "1.0"
    tags: Optional[List[str]] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class SkillRegistry:
    """Skill 注册表 - 动态存储和检索 Skills"""

    def __init__(self):
        self._skills: Dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        """注册 Skill"""
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name} (source: {skill.source})")

    def get(self, name: str) -> Optional[SkillDefinition]:
        """获取 Skill by name"""
        # 支持多种名称格式
        return self._skills.get(name) or \
               self._skills.get(name.lower()) or \
               self._skills.get(name.title())

    def list_skills(self) -> List[SkillDefinition]:
        """列出所有 Skills"""
        return sorted(self._skills.values(), key=lambda s: s.name)

    def list_by_tag(self, tag: str) -> List[SkillDefinition]:
        """按标签筛选 Skills"""
        return [s for s in self._skills.values() if tag in s.tags]

    def list_by_source(self, source: str) -> List[SkillDefinition]:
        """按来源筛选 Skills"""
        return [s for s in self._skills.values() if s.source == source]

    def unregister(self, name: str) -> bool:
        """取消注册 Skill"""
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def clear(self) -> None:
        """清空注册表"""
        self._skills.clear()


class SkillLoader:
    """Skill 动态加载器"""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def load_from_dir(self, directory: Path, source: str = "user") -> List[SkillDefinition]:
        """从目录加载 Skills

        目录结构:
        - <root>/<skill-dir>/SKILL.md
        - <root>/<skill-dir>/skill.yaml (可选配置)
        """
        skills = []
        if not directory.exists():
            logger.warning(f"Skills directory not found: {directory}")
            return skills

        for child in sorted(directory.iterdir()):
            if child.is_dir():
                skill_md = child / "SKILL.md"
                skill_yaml = child / "skill.yaml"

                if skill_md.exists():
                    content = skill_md.read_text(encoding="utf-8")
                    default_name = child.name

                    # 解析 YAML 配置（如果存在）
                    config = {}
                    if skill_yaml.exists():
                        try:
                            import yaml
                            config = yaml.safe_load(skill_yaml.read_text(encoding="utf-8")) or {}
                        except Exception as e:
                            logger.warning(f"Failed to parse {skill_yaml}: {e}")

                    # 解析 Markdown frontmatter
                    name, description, tags = self._parse_skill_markdown(default_name, content, config)

                    skill = SkillDefinition(
                        name=name,
                        description=description,
                        content=content,
                        source=source,
                        path=str(skill_md),
                        version=config.get("version", "1.0"),
                        tags=tags
                    )
                    skills.append(skill)
                    self.registry.register(skill)

        return skills

    def load_from_file(self, file_path: Path, source: str = "user") -> Optional[SkillDefinition]:
        """从单个 Markdown 文件加载 Skill"""
        if not file_path.exists():
            return None

        content = file_path.read_text(encoding="utf-8")
        default_name = file_path.stem

        name, description, tags = self._parse_skill_markdown(default_name, content, {})

        skill = SkillDefinition(
            name=name,
            description=description,
            content=content,
            source=source,
            path=str(file_path),
            tags=tags
        )
        self.registry.register(skill)
        return skill

    def _parse_skill_markdown(self, default_name: str, content: str, config: dict) -> tuple:
        """解析 Skill Markdown 文件"""
        name = config.get("name", default_name)
        description = config.get("description", "")
        tags = config.get("tags", [])

        lines = content.splitlines()

        # 尝试解析 YAML frontmatter
        if content.startswith("---"):
            end_idx = content.find("\n---", 4)
            if end_idx != -1:
                try:
                    import yaml
                    frontmatter = yaml.safe_load(content[4:end_idx])
                    if isinstance(frontmatter, dict):
                        if frontmatter.get("name"):
                            name = frontmatter["name"]
                        if frontmatter.get("description"):
                            description = frontmatter["description"]
                        if frontmatter.get("tags"):
                            tags = frontmatter.get("tags", [])
                except Exception:
                    pass

        # 从标题和第一段提取
        if not description:
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("# "):
                    if name == default_name:
                        name = stripped[2:].strip()
                    continue
                if stripped and not stripped.startswith("---") and not stripped.startswith("#"):
                    description = stripped[:200]
                    break

        if not description:
            description = f"Skill: {name}"

        return name, description, tags


def load_skill_registry(
    cwd: Optional[Path] = None,
    extra_skill_dirs: Optional[Iterable[Path]] = None,
) -> SkillRegistry:
    """加载 Skill 注册表（兼容新旧目录结构）"""
    registry = SkillRegistry()
    loader = SkillLoader(registry)

    # 加载 bundled skills
    bundled_dir = Path(__file__).parent / "bundled"
    loader.load_from_dir(bundled_dir, source="bundled")

    # 加载项目根 skills/ 目录（新版 scoped 结构）
    skills_root = Path(__file__).parent.parent.parent / "skills"
    if skills_root.exists():
        for scope_dir_name in ("main", "user", "downloaded", "skill-management"):
            scope_path = skills_root / scope_dir_name
            if scope_path.exists() and scope_path.is_dir():
                loader.load_from_dir(scope_path, source=f"scoped:{scope_dir_name}")

    # 加载旧版用户 skills（向后兼容）
    user_dir = skills_root / "user"
    if user_dir.exists():
        loader.load_from_dir(user_dir, source="user")

    # 加载额外目录
    if extra_skill_dirs:
        for dir_path in extra_skill_dirs:
            loader.load_from_dir(Path(dir_path), source="extra")

    # 加载项目目录 skills
    if cwd:
        project_skills = cwd / ".openharness" / "skills"
        if project_skills.exists():
            loader.load_from_dir(project_skills, source="project")

    return registry


# ============================================
# 预定义的 BugFix Skills
# ============================================

BUNDLED_SKILLS = {
    "code_review": """---
name: Code Review
description: Review code for potential bugs and issues
tags: [review, analysis, bugs]
---

# Code Review Skill

You are a Code Review Expert. Your task is to analyze code and identify potential bugs.

## Review Process

1. **Static Analysis**
   - Check for syntax errors
   - Check for logical errors
   - Check for potential runtime errors

2. **Pattern Detection**
   - Identify common bug patterns
   - Check for edge cases
   - Analyze error handling

3. **Output Format**
   - Bug location (file, line)
   - Bug type (syntax, logic, runtime)
   - Severity (critical, major, minor)
   - Description

## Example

Input: Java code with break statement outside loop
Output: Bug found at line 46 - break statement not in loop/switch context
""",

    "bug_analysis": """---
name: Bug Analysis
description: Deep analysis of bug root cause
tags: [analysis, root-cause, diagnosis]
---

# Bug Analysis Skill

You are a Bug Analysis Expert. Analyze the root cause of identified bugs.

## Analysis Steps

1. **Understand Context**
   - Read surrounding code
   - Understand the intended behavior
   - Check dependencies

2. **Root Cause Analysis**
   - Identify the exact cause
   - Trace the bug propagation path
   - Analyze impact scope

3. **Fix Strategy**
   - Propose fix approaches
   - Evaluate each approach
   - Select best strategy

## Output Format

- Root cause description
- Impact analysis
- Recommended fix strategy
""",

    "code_fix": """---
name: Code Fix
description: Generate patches to fix identified bugs
tags: [fix, patch, repair]
---

# Code Fix Skill

You are a Code Fix Expert. Generate patches to repair bugs.

## Fix Process

1. **Understand Bug**
   - Read bug analysis
   - Understand the fix strategy
   - Check constraints

2. **Generate Patch**
   - Write the corrected code
   - Ensure minimal changes
   - Maintain code style

3. **Validate Fix**
   - Check syntax correctness
   - Verify logic correctness
   - Test edge cases

## Patch Format

```diff
--- original
+++ fixed
@@ line_number @@
- buggy_code
+ fixed_code
```
""",

    "validation": """---
name: Validation
description: Validate bug fixes through testing
tags: [validation, testing, verification]
---

# Validation Skill

You are a Validation Expert. Verify that bug fixes are correct.

## Validation Steps

1. **Syntax Check**
   - Compile/parse the fixed code
   - Check for new errors

2. **Logic Check**
   - Verify the fix addresses the bug
   - Check for side effects

3. **Test Execution**
   - Run relevant tests
   - Check test coverage

## Output Format

- Validation result (pass/fail)
- Issues found (if any)
- Confidence level
""",

    "discussion": """---
name: Agent Discussion
description: Protocol for multi-agent collaboration
tags: [collaboration, discussion, consensus]
---

# Agent Discussion Skill

Protocol for agents to discuss and reach consensus.

## Discussion Format

1. **Initial Position**
   - Agent A: Present initial analysis
   - Include: findings, reasoning, confidence

2. **Response**
   - Agent B: Review and respond
   - Agree or provide alternative view

3. **Consensus**
   - Both agents agree on final position
   - Record: agreed conclusion, confidence

## Example

Reviewer: "Found bug at line 46 - break outside loop"
Analyzer: "Agree. Root cause is misplaced break statement"
Consensus: "Bug confirmed at line 46"
"""
}


def get_bundled_skills() -> List[SkillDefinition]:
    """获取内置 Skills"""
    skills = []
    for name, content in BUNDLED_SKILLS.items():
        skill = SkillDefinition(
            name=name,
            description=f"Bundled skill: {name}",
            content=content,
            source="bundled",
            tags=["bundled"]
        )
        skills.append(skill)
    return skills