#!/usr/bin/env python
"""
Standalone Skill Download Script
=================================

PDF Harness Engineering pattern: Stage 3 — Agent executes this script
to download a skill from a URL, extract it, validate the SKILL.md,
and persist to ChromaDB + Redis.

Usage (called by Agent):
    python download_skill.py <URL> [--name SKILL_NAME] [--scope main|user|downloaded]

Example:
    python download_skill.py https://example.com/skills/concurrency-checker.zip
    python download_skill.py https://example.com/skills/memory-leak.md --name memory-leak-detector
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.skills.manager import SkillManager, get_skill_manager
from agent.skills.progressive import SkillScope


async def main():
    parser = argparse.ArgumentParser(description="Download a skill from URL")
    parser.add_argument("url", help="URL to download (zip or .md)")
    parser.add_argument("--name", dest="skill_name", default=None, help="Custom skill name")
    parser.add_argument("--scope", default="downloaded", choices=["main", "user", "downloaded"],
                        help="Target scope (default: downloaded)")
    args = parser.parse_args()

    try:
        scope = SkillScope(args.scope)
    except ValueError:
        print(f"ERROR: Invalid scope '{args.scope}'")
        sys.exit(1)

    print(f"[download_skill] Downloading from: {args.url}")
    print(f"[download_skill] Target scope: {scope.value}")

    mgr = get_skill_manager()
    fm = await mgr.download_skill(args.url, scope, args.skill_name)

    if fm is None:
        print("ERROR: Download failed")
        sys.exit(1)

    print(f"\nDownload successful!")
    print(f"  Name:        {fm.name}")
    print(f"  Description: {fm.description}")
    print(f"  Version:     {fm.version}")
    print(f"  Scope:       {fm.scope.value}")
    print(f"  Path:        {fm.path}")
    print(f"  Tags:        {', '.join(fm.tags) if fm.tags else 'none'}")

    # Print JSON for Agent consumption
    import json
    print(f"\n[JSON_RESULT]")
    print(json.dumps(fm.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
