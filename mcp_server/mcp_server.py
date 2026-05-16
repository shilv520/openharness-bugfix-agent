#!/usr/bin/env python
"""
Bug Fix MCP Server - 接入真实 Multi-Agent 协同系统

提供工具:
- analyze_code: Reviewer + Analyzer 协同分析
- generate_patch: Fixer 生成补丁
- run_test: Validator 验证修复
- query_defects4j: 查询 Defects4J 数据集
- full_bugfix: 完整 Multi-Agent 协同修复流程

启动: python mcp_server.py --stdio
"""

import asyncio
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.collaborative_agent import (
    CollaborativeReviewerAgent,
    CollaborativeAnalyzerAgent,
    CollaborativeFixerAgent,
    CollaborativeValidatorAgent,
    call_llm
)
from agent.communication import get_comm_bus, DiscussionProtocol, FeedbackProtocol, reset_comm_bus


class BugFixMcpServer:
    """Bug Fix MCP Server - 真实接入 Multi-Agent 协同"""

    def __init__(self):
        self.name = "openharness-bugfix-mcp"
        self._agents = None
        self._comm_bus = None
        # Reset global comm bus to avoid cross-test interference
        reset_comm_bus()

        self.tools = [
            {
                "name": "analyze_code",
                "description": "Reviewer+Analyzer协同分析代码，定位Bug根因",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "源代码"},
                        "language": {"type": "string", "description": "编程语言 java/python"}
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "generate_patch",
                "description": "Fixer生成修复补丁",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bug_location": {"type": "string", "description": "Bug位置"},
                        "bug_type": {"type": "string", "description": "Bug类型"},
                        "root_cause": {"type": "string", "description": "根因分析"},
                        "code": {"type": "string", "description": "原始代码"}
                    },
                    "required": ["bug_location", "code"]
                }
            },
            {
                "name": "run_test",
                "description": "Validator验证修复",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fixed_code": {"type": "string", "description": "修复后代码"},
                        "original_code": {"type": "string", "description": "原始代码"},
                        "patch": {"type": "string", "description": "补丁"}
                    },
                    "required": ["fixed_code"]
                }
            },
            {
                "name": "query_defects4j",
                "description": "查询Defects4J数据集",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string", "description": "项目 Math/Lang/Chart"},
                        "bug_id": {"type": "number", "description": "Bug编号"}
                    },
                    "required": ["project"]
                }
            },
            {
                "name": "full_bugfix",
                "description": "完整Multi-Agent协同修复流程",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "源代码"},
                        "language": {"type": "string", "description": "编程语言"}
                    },
                    "required": ["code"]
                }
            }
        ]

    async def _get_agents(self):
        """懒加载 Agent"""
        if self._agents is None:
            self._comm_bus = get_comm_bus()
            self._agents = {
                "reviewer": CollaborativeReviewerAgent(),
                "analyzer": CollaborativeAnalyzerAgent(),
                "fixer": CollaborativeFixerAgent(),
                "validator": CollaborativeValidatorAgent(),
            }
        return self._agents

    async def handle_tool_call(self, tool_name: str, arguments: dict) -> dict:
        """处理工具调用 - 真实执行"""
        agents = await self._get_agents()

        # 设置分层记忆会话
        from datetime import datetime
        session_id = f"mcp_{tool_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        for agent in agents.values():
            agent.set_memory_session("mcp_user", session_id)

        if tool_name == "analyze_code":
            return await self._analyze_code(agents, arguments)

        elif tool_name == "generate_patch":
            return await self._generate_patch(agents, arguments)

        elif tool_name == "run_test":
            return await self._run_test(agents, arguments)

        elif tool_name == "query_defects4j":
            return await self._query_defects4j(arguments)

        elif tool_name == "full_bugfix":
            return await self._full_bugfix(agents, arguments)

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _analyze_code(self, agents: dict, args: dict) -> dict:
        """Reviewer + Analyzer 协同分析"""
        code = args.get("code", "")
        language = args.get("language", "java")

        reviewer = agents["reviewer"]
        analyzer = agents["analyzer"]

        # Step 1: Reviewer 审查
        context = {"code": code, "language": language, "task": "analyze_code"}
        review_result = await reviewer.execute_action("review_code", context)

        bugs = review_result.get("bug_candidates", [])

        # Step 2: Reviewer-Analyzer 讨论
        if bugs:
            discussion = await reviewer.collaborate(
                "Analyzer", "Bug确认",
                {"view": f"发现{len(bugs)}个潜在Bug", "candidates": bugs}
            )
        else:
            discussion = {"agreed": False, "consensus": {}}

        # Step 3: Analyzer 分析
        analysis_context = {
            "code": code,
            "review_result": review_result,
            "discussion": discussion
        }
        analysis_result = await analyzer.execute_action("analyze_bug", analysis_context)

        return {
            "review": {
                "bugs_found": len(bugs),
                "candidates": bugs,
                "code_quality": review_result.get("code_quality", "unknown")
            },
            "discussion": {
                "agreed": discussion.get("agreed", False),
                "consensus": discussion.get("consensus", {})
            },
            "analysis": {
                "bug_location": analysis_result.get("bug_location", "unknown"),
                "bug_type": analysis_result.get("bug_type", "unknown"),
                "root_cause": analysis_result.get("root_cause", "unknown"),
                "fix_suggestion": analysis_result.get("fix_suggestion", "unknown"),
                "confidence": analysis_result.get("confidence", 0.5)
            }
        }

    async def _generate_patch(self, agents: dict, args: dict) -> dict:
        """Fixer 生成补丁"""
        analysis_result = {
            "bug_location": args.get("bug_location", ""),
            "bug_type": args.get("bug_type", ""),
            "root_cause": args.get("root_cause", ""),
            "fix_suggestion": args.get("fix_suggestion", "")
        }
        code = args.get("code", "")

        fixer = agents["fixer"]

        context = {
            "code": code,
            "analysis_result": analysis_result,
            "fix_discussion": {"consensus": {"fix_approach": args.get("fix_suggestion", "")}}
        }

        result = await fixer.execute_action("generate_patch", context)

        return {
            "patch": result.get("patch", ""),
            "fixed_code": result.get("fixed_code", ""),
            "explanation": result.get("fix_explanation", ""),
            "confidence": result.get("confidence", 0.5)
        }

    async def _run_test(self, agents: dict, args: dict) -> dict:
        """Validator 验证修复"""
        fixed_code = args.get("fixed_code", "")
        original_code = args.get("original_code", "")
        patch = args.get("patch", "")

        validator = agents["validator"]

        context = {
            "fixed_code": fixed_code,
            "code": original_code,
            "patch": patch
        }

        result = await validator.execute_action("validate_fix", context)

        return {
            "test_passed": result.get("test_passed", False),
            "validation_reason": result.get("validation_reason", ""),
            "side_effects": result.get("side_effects", []),
            "quality_score": result.get("quality_score", 0),
            "confidence": result.get("confidence", 0.5)
        }

    async def _query_defects4j(self, args: dict) -> dict:
        """查询 Defects4J 数据集"""
        project = args.get("project", "Math")
        bug_id = args.get("bug_id", 1)

        # 真实读取 Defects4J 数据
        defects4j_dir = Path(__file__).parent.parent / "data" / "defects4j"
        project_path = defects4j_dir / "framework" / "projects" / project

        result = {
            "project": project,
            "bug_id": bug_id,
            "exists": False,
            "info": {}
        }

        if project_path.exists():
            result["exists"] = True

            # 读取活跃 bugs
            active_bugs_file = project_path / "active-bugs.csv"
            if active_bugs_file.exists():
                bugs_content = active_bugs_file.read_text()
                result["info"]["available_bugs"] = bugs_content.strip()[:500]

            # 读取 bug 元数据（如果有）
            metadata_dir = defects4j_dir / "framework" / "projects" / project / "modified_classes"
            if metadata_dir.exists():
                result["info"]["has_modified_classes"] = True

            # 尝试读取具体 Bug 的补丁
            patch_path = project_path / "patches" / f"{bug_id}.src.patch"
            if patch_path.exists():
                patch_content = patch_path.read_text()
                result["info"]["patch_preview"] = patch_content[:500]
                result["has_patch"] = True
            else:
                result["has_patch"] = False

        # 尝试查找 git log 中的 bug 信息
        project_repos = defects4j_dir / "project_repos"
        if project_repos.exists():
            repos = [r.name for r in project_repos.iterdir() if r.is_dir()]
            result["available_repos"] = repos

        return result

    async def _full_bugfix(self, agents: dict, args: dict) -> dict:
        """完整 Multi-Agent 协同修复流程"""
        code = args.get("code", "")
        language = args.get("language", "java")

        reviewer = agents["reviewer"]
        analyzer = agents["analyzer"]
        fixer = agents["fixer"]
        validator = agents["validator"]

        steps = []

        # Step 1: Reviewer 审查
        review_result = await reviewer.execute_action("review_code", {"code": code, "language": language})
        steps.append({"step": "review", "result": "success", "bugs_found": len(review_result.get("bug_candidates", []))})

        # Step 2: Reviewer-Analyzer 讨论
        if review_result.get("bug_candidates"):
            discussion_ra = await reviewer.collaborate(
                "Analyzer", "Bug确认",
                {"view": f"发现Bug", "candidates": review_result["bug_candidates"]}
            )
            steps.append({"step": "discuss_review_analyzer", "agreed": discussion_ra.get("agreed", False)})
        else:
            steps.append({"step": "no_bugs_found", "result": "skip"})
            return {"success": False, "message": "No bugs found", "steps": steps}

        # Step 3: Analyzer 分析
        analysis_result = await analyzer.execute_action("analyze_bug", {
            "code": code,
            "review_result": review_result
        })
        steps.append({"step": "analyze", "root_cause": analysis_result.get("root_cause", "")[:100]})

        # Step 4: Analyzer-Fixer 讨论
        discussion_af = await analyzer.collaborate(
            "Fixer", "修复策略",
            {"view": f"Bug: {analysis_result.get('bug_type')}", "analysis": analysis_result}
        )
        steps.append({"step": "discuss_analyzer_fixer", "agreed": discussion_af.get("agreed", False)})

        # Step 5: Fixer 生成补丁
        fix_result = await fixer.execute_action("generate_patch", {
            "code": code,
            "analysis_result": analysis_result,
            "fix_discussion": discussion_af
        })
        steps.append({"step": "fix", "patch_generated": bool(fix_result.get("patch"))})

        # Step 6: Validator 最终验证
        validation_result = await validator.execute_action("validate_fix", {
            "fixed_code": fix_result.get("fixed_code", ""),
            "code": code,
            "patch": fix_result.get("patch", "")
        })
        steps.append({"step": "validate", "passed": validation_result.get("test_passed", False)})

        return {
            "success": validation_result.get("test_passed", False),
            "steps": steps,
            "bug_location": analysis_result.get("bug_location", ""),
            "bug_type": analysis_result.get("bug_type", ""),
            "root_cause": analysis_result.get("root_cause", ""),
            "fix_suggestion": analysis_result.get("fix_suggestion", ""),
            "patch": fix_result.get("patch", ""),
            "fixed_code": fix_result.get("fixed_code", ""),
            "test_passed": validation_result.get("test_passed", False),
            "total_steps": len(steps)
        }

    async def handle_request(self, request: dict) -> dict:
        """处理 MCP JSON-RPC 请求（异步）"""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": self.name,
                        "version": "2.0.0",
                        "description": "Bug Fix Multi-Agent System with DeepSeek V4 Pro"
                    }
                }
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.tools}
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            try:
                result = await asyncio.wait_for(
                    self.handle_tool_call(tool_name, arguments),
                    timeout=300.0
                )
            except asyncio.TimeoutError:
                result = {"error": "Tool call timed out (120s)"}
            except Exception as e:
                result = {"error": f"Tool execution failed: {str(e)}"}

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2, ensure_ascii=False)
                    }]
                }
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"}
            }

    async def run_stdio(self):
        """stdio 模式运行"""
        import logging
        logging.basicConfig(level=logging.WARNING)

        print(json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }), flush=True)

        loop = asyncio.get_event_loop()

        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                request = json.loads(line)
                response = await self.handle_request(request)

                print(json.dumps(response, ensure_ascii=False), flush=True)

            except json.JSONDecodeError:
                error_resp = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error"}
                }
                print(json.dumps(error_resp), flush=True)
            except Exception as e:
                error_resp = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                }
                print(json.dumps(error_resp, ensure_ascii=False), flush=True)


async def main():
    server = BugFixMcpServer()

    if "--stdio" in sys.argv:
        await server.run_stdio()
    else:
        # 测试模式
        print("=" * 60)
        print("Bug Fix MCP Server (v2.0 - Real Multi-Agent)")
        print("=" * 60)
        print("\n可用工具:")
        for tool in server.tools:
            print(f"  - {tool['name']}: {tool['description']}")
        print("\n启动: python mcp_server.py --stdio")


if __name__ == "__main__":
    asyncio.run(main())