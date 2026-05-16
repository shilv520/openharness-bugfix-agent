"""
AgentDefinition — YAML驱动的子Agent定义
=========================================

PDF Harness Engineering: 每个子Agent由YAML配置文件定义
  - name / description / system_prompt
  - capabilities / tools / skills
  - output_schema (结构化输出规范)

YAML位置: config/agents/*.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger("delegation.definition")


@dataclass
class AgentDefinition:
    """子Agent模板定义（从YAML加载）。

    对应PDF项目的Agent Definition YAML配置。
    """

    name: str
    description: str
    system_prompt: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    output_schema: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return f"[{self.name}] v{self.version}"

    @property
    def brief(self) -> str:
        """一行摘要（用于发现阶段，节省Token）"""
        tags_str = ", ".join(self.tags[:3])
        return f"{self.name} v{self.version} — {self.description} (tags: {tags_str})"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "skills": self.skills,
            "output_schema": self.output_schema,
        }

    def to_prompt_fragment(self) -> str:
        """生成注入到主Agent system prompt的描述片段"""
        tools_str = ", ".join(self.tools) if self.tools else "无"
        skills_str = ", ".join(self.skills) if self.skills else "无"
        return (
            f"### {self.name}\n"
            f"- 描述: {self.description}\n"
            f"- 能力: {', '.join(self.capabilities)}\n"
            f"- 工具: {tools_str}\n"
            f"- 技能: {skills_str}\n"
        )


# ── YAML 加载 ─────────────────────────────────────────────────

def _get_config_dir() -> Path:
    """获取agent配置目录"""
    # 从当前文件位置推算项目根
    this_dir = Path(__file__).resolve().parent  # agent/delegation/
    project_root = this_dir.parent.parent  # 项目根
    config_dir = project_root / "config" / "agents"
    return config_dir


def load_agent_definition(yaml_path: str | Path) -> Optional[AgentDefinition]:
    """从YAML文件加载单个Agent定义。

    Args:
        yaml_path: YAML文件路径

    Returns:
        AgentDefinition or None (解析失败)
    """
    path = Path(yaml_path)
    if not path.exists():
        logger.warning(f"[AgentDef] File not found: {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "name" not in data:
            logger.warning(f"[AgentDef] Invalid YAML (missing 'name'): {path}")
            return None

        return AgentDefinition(
            name=data["name"],
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt", ""),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            capabilities=data.get("capabilities", []),
            tools=data.get("tools", []),
            skills=data.get("skills", []),
            output_schema=data.get("output_schema", {}),
        )
    except Exception as e:
        logger.error(f"[AgentDef] Failed to load {path}: {e}")
        return None


def load_all_definitions(config_dir: str | Path = None) -> List[AgentDefinition]:
    """加载所有Agent定义。

    Args:
        config_dir: YAML配置目录，默认为 config/agents/

    Returns:
        所有成功加载的AgentDefinition列表
    """
    directory = Path(config_dir) if config_dir else _get_config_dir()

    if not directory.exists():
        logger.warning(f"[AgentDef] Config directory not found: {directory}")
        return _get_fallback_definitions()

    definitions = []
    for yaml_file in sorted(directory.glob("*.yaml")):
        ad = load_agent_definition(yaml_file)
        if ad:
            definitions.append(ad)
            logger.info(f"[AgentDef] Loaded: {ad.display_name} from {yaml_file.name}")

    if not definitions:
        logger.warning("[AgentDef] No definitions loaded, using fallbacks")
        return _get_fallback_definitions()

    return definitions


def _get_fallback_definitions() -> List[AgentDefinition]:
    """当YAML文件不可用时的硬编码回退定义"""
    return [
        AgentDefinition(
            name="code-reviewer",
            description="代码审查专家 — 识别潜在Bug和代码质量问题",
            system_prompt="你是一位资深代码审查专家。请审查给定的代码，找出所有潜在Bug、安全漏洞和代码质量问题。输出JSON格式。",
            capabilities=["静态代码分析", "Bug模式识别", "安全漏洞检测"],
            tools=["read_file", "grep_search", "glob_search"],
            skills=["code-review"],
            output_schema={"bug_candidates": "array", "severity": "string", "summary": "string"},
        ),
        AgentDefinition(
            name="bug-analyzer",
            description="Bug根因分析专家 — 深入分析Bug根因和修复策略",
            system_prompt="你是一位Bug根因分析专家。请分析审查发现的每个Bug，判定真假阳性，追溯根因，评估影响范围，提出修复策略。输出JSON格式。",
            capabilities=["根因追溯", "Bug分类", "影响评估", "修复方案对比"],
            tools=["read_file", "grep_search", "glob_search"],
            skills=["bug-analysis"],
            output_schema={"confirmed_bugs": "array", "root_cause_summary": "string", "fix_strategy": "string"},
        ),
        AgentDefinition(
            name="code-fixer",
            description="代码修复专家 — 生成精确、安全、最小化的补丁",
            system_prompt="你是一位代码修复专家。请根据根因分析结果生成精确的代码补丁。遵循最小化修改原则，保持代码风格，添加必要的防御性编程。输出JSON格式含patch和fixed_code。",
            capabilities=["精确补丁生成", "代码重构", "防御性编程"],
            tools=["read_file", "write_file", "edit_file", "grep_search"],
            skills=["fix-patterns"],
            output_schema={"patch": "string", "fixed_code": "string", "changes": "array"},
        ),
        AgentDefinition(
            name="test-validator",
            description="测试验证专家 — 验证修复正确性，检查副作用",
            system_prompt="你是一位测试验证专家。请验证修复后的代码是否正确，检查是否引入新问题，确认所有Bug是否都已修复。输出JSON格式含test_passed和验证详情。",
            capabilities=["测试验证", "副作用检测", "代码质量确认"],
            tools=["read_file", "grep_search", "exec_python", "exec_shell"],
            skills=["test-generation"],
            output_schema={"test_passed": "boolean", "bug_results": "array", "new_issues": "array"},
        ),
    ]
