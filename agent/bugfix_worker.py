"""
Bug Fix Agent Worker - 启动入口
===============================

启动Bug修复Agent Worker服务
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bugfix-worker")

async def main():
    """主函数"""

    logger.info("=" * 60)
    logger.info("Bug Fix Agent Worker 启动")
    logger.info("=" * 60)

    # 测试Plan-Execute状态图
    from graph.plan_execute import run_bugfix

    test_code = """
public class BigFraction {
    public BigFraction(double value) {
        if (epsilon == 0.0 && FastMath.abs(q1) < maxDenominator) {
            break;
        }
        throw new FractionConversionException(value, p2, q2);
    }
}
"""

    logger.info("测试Bug修复流程...")
    result = await run_bugfix(test_code, "java")

    logger.info(f"Bug位置: {result.get('bug_location')}")
    logger.info(f"测试通过: {result.get('test_passed')}")
    logger.info(f"重规划次数: {result.get('replan_count')}")


if __name__ == "__main__":
    asyncio.run(main())