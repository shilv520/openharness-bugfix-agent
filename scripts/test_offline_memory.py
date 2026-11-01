"""离线 memory（做梦）机制自测 —— 不依赖 Redis/ChromaDB，纯 fallback 内存态。"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.skills.progressive import SkillFrontmatter, SkillScope
from agent.skills.persistence import SkillPersistence
from agent.skills.manager import SkillManager
from agent.skills.offline_memory import SkillDreamer


def make_persist():
    """绕过 __init__ 的网络连接，构造纯 fallback 的持久层。"""
    p = SkillPersistence.__new__(SkillPersistence)
    p._redis = None
    p._collection = None
    p._chroma_client = None
    p._fallback = {}
    p._memory_fallback = {}
    return p


def fm(name, created=None, last_hit=""):
    f = SkillFrontmatter(name=name, description=f"skill {name}", scope=SkillScope.USER)
    f.created_at = created
    f.updated_at = created
    f.last_hit_at = last_hit
    f.path = ""
    return f


def main():
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(hours=2)).isoformat()
    stale = (now - timedelta(days=30)).isoformat()

    persist = make_persist()
    with TemporaryDirectory() as tmp:
        mgr = SkillManager(skills_root=Path(tmp), persistence=persist)

        # ── 造数据 ──
        persist.store_skill(fm("hot-skill", created=recent, last_hit=recent))   # 高频 → keep
        persist.store_skill(fm("stale-skill", created=stale))                    # 低频 → archive
        persist.store_skill(fm("warm-again", created=stale))                     # 先归档，再恢复
        mgr.archive_skill("warm-again")
        persist.record_usage("warm-again")  # 归档后又被命中 → last_hit 变 recent

        # ── 执行离线 dream ──
        report = SkillDreamer(mgr).dream(now=now)

        print("summary :", report["summary"])
        print("archived:", report["archived"])
        print("restored:", report["restored"])
        print("kept    :", report["kept"])
        print("memory  :", report["memory_key"])

        # ── 断言 ──
        assert report["archived"] == ["stale-skill"], report["archived"]
        assert report["restored"] == ["warm-again"], report["restored"]
        assert "hot-skill" in report["kept"], report["kept"]

        assert persist.get_skill_meta("stale-skill").scope == SkillScope.ARCHIVED
        assert persist.get_skill_meta("stale-skill").archived_from == "user"
        assert persist.get_skill_meta("warm-again").scope == SkillScope.USER
        assert persist.get_skill_meta("warm-again").archived_from == ""
        assert persist.get_skill_meta("hot-skill").scope == SkillScope.USER

        # 默认发现排除 archived；include_archived=True 才可见
        default_names = {f.name for f in mgr.list_frontmatters()}
        all_names = {f.name for f in mgr.list_frontmatters(include_archived=True)}
        assert "stale-skill" not in default_names, default_names
        assert "stale-skill" in all_names, all_names

        # dream 决策报告已写入 memory（用 memory 兜底，可审计）
        mem = persist.get_memory(report["memory_key"])
        assert mem is not None and mem["archived"] == ["stale-skill"], mem

        # record_usage 统计被正确累计
        assert persist.get_usage("warm-again")["hit_count"] == 1
        assert persist.get_usage("warm-again")["last_hit_at"]

        print("\n[OK] ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
