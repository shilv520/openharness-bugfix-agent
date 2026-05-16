"""
draw_graph: 将 LangGraph 编译后的状态图渲染为 PNG。
参考项目使用 graphviz，这里用纯 Python (PIL) 实现，无外部依赖。
"""

import os
from collections import defaultdict


def draw_graph(graph_obj, output_path: str):
    """
    从编译后的 LangGraph 图生成 PNG 可视化。

    先尝试 graphviz (dot) → PNG，失败则用 PIL 纯 Python 渲染。
    """
    nodes, edges = _extract_graph(graph_obj)

    # 尝试 graphviz
    if _try_graphviz(nodes, edges, output_path):
        return

    # 回退: PIL 纯 Python
    _try_pil(nodes, edges, output_path)


def _extract_graph(graph_obj):
    """从编译图中提取 nodes 和 edges"""
    try:
        raw = graph_obj.get_graph()
        nodes = list(raw.nodes.keys()) if hasattr(raw.nodes, 'keys') else list(raw.nodes)
        edges = [(e.source, e.target) for e in raw.edges]
        return nodes, edges
    except Exception:
        return [], []


def _try_graphviz(nodes, edges, output_path) -> bool:
    """用 graphviz 渲染，成功返回 True"""
    try:
        import graphviz  # noqa: F811
        # 尝试找 dot
        dot_paths = [
            "dot",
            "C:/Program Files/Graphviz/bin/dot.exe",
            "C:/Program Files (x86)/Graphviz/bin/dot.exe",
        ]
        dot = None
        for p in dot_paths:
            if os.path.isfile(p) or p == "dot":
                dot = p
                break

        if dot is None:
            return False

        from graphviz import Digraph
        g = Digraph(format="png")
        g.attr(rankdir="TB")

        for n in nodes:
            g.node(n, n, shape="box", style="filled,rounded")
        for s, t in edges:
            g.edge(s, t)

        g.render(output_path.replace(".png", ""), cleanup=True)
        print(f"draw_graph (graphviz): {output_path}")
        return True
    except Exception:
        return False


def _try_pil(nodes, edges, output_path):
    """用 PIL 纯 Python 渲染"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # PIL 也没有，输出 Mermaid 回退
        _mermaid_fallback(nodes, edges, output_path.replace(".png", ".md"))
        return

    # 布局参数
    n = len(nodes)
    if n == 0:
        return

    box_w, box_h = 160, 50
    margin_x, margin_y = 60, 40
    spacing_x, spacing_y = 60, 80

    # 计算列/行排列 (优先水平排列)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    img_w = margin_x * 2 + cols * box_w + (cols - 1) * spacing_x
    img_h = margin_y * 2 + rows * box_h + (rows - 1) * spacing_y + 60

    # 节点位置
    pos = {}
    for i, name in enumerate(nodes):
        col = i % cols
        row = i // cols
        x = margin_x + col * (box_w + spacing_x)
        y = margin_y + row * (box_h + spacing_y)
        pos[name] = (x, y)

    # 颜色映射
    def node_color(name):
        n = name.lower()
        if "start" in n:
            return ("#ECEFF1", "#546E7A")
        if "end" in n:
            return ("#ECEFF1", "#37474F")
        if "review" in n:
            return ("#E3F2FD", "#1565C0")
        if "analy" in n:
            return ("#E8F5E9", "#2E7D32")
        if "fix" in n:
            return ("#FFF8E1", "#F9A825")
        if "valid" in n:
            return ("#F3E5F5", "#6A1B9A")
        if "replan" in n:
            return ("#FFEBEE", "#C62828")
        if "plan" in n:
            return ("#FFF3E0", "#E65100")
        if "execut" in n:
            return ("#E8F5E9", "#2E7D32")
        return ("#FAFAFA", "#9E9E9E")

    img = Image.new("RGB", (img_w, img_h), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # 画边
    edge_set = set()
    for s, t in edges:
        if s not in pos or t not in pos:
            continue
        edge_set.add((s, t))
        sx, sy = pos[s]
        tx, ty = pos[t]
        cx1 = sx + box_w
        cy1 = sy + box_h // 2
        cx2 = tx
        cy2 = ty + box_h // 2
        # 简单直角连线
        mx = (cx1 + cx2) // 2
        draw.line([(cx1, cy1), (mx, cy1), (mx, cy2), (cx2, cy2)],
                  fill="#888888", width=2)
        # 箭头
        draw.polygon([(cx2, cy2), (cx2 - 8, cy2 - 4), (cx2 - 8, cy2 + 4)],
                     fill="#888888")

    # 画节点
    for name, (x, y) in pos.items():
        fill, border = node_color(name)
        draw.rounded_rectangle([x, y, x + box_w, y + box_h],
                               radius=10, fill=fill, outline=border, width=2)

        # 文字
        label = name.replace("__", "").replace("_", " ").title()
        if label.upper() in ("START", "END"):
            label = label.upper()

        # 估算文字位置 (PIL 简单文字)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x + (box_w - tw) // 2
        ty = y + (box_h - th) // 2
        draw.text((tx, ty), label, fill="#263238", font=font)

    img.save(output_path, "PNG")
    print(f"draw_graph (PIL): {output_path}")


def _mermaid_fallback(nodes, edges, output_path: str):
    """回退: 输出 Mermaid 文本"""
    lines = ["# LangGraph Workflow\n", "```mermaid", "flowchart TD"]
    for n in nodes:
        label = n.replace("__", "").replace("_", " ").title()
        if label.upper() == "START":
            lines.append("    START([START])")
        elif label.upper() == "END":
            lines.append("    END1([END])")
        else:
            lines.append(f"    {n}[\"{label}\"]")
    for s, t in edges:
        lines.append(f"    {s} --> {t}")
    lines.append("```")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"draw_graph (mermaid fallback): {output_path}")
