#!/usr/bin/env python3
"""smoke_damo_api.py — Task 1.7 平台 API 基线的最小连通性冒烟（1 次调用）。

在实例上跑：/root/miniconda3/bin/python3 /root/rl_training/scripts/smoke_damo_api.py
安全纪律：key 只从 /root/.damo_key 文件读（chmod 600，repo 之外），
永不进 argv / 永不打印明文；本脚本打印 HTTP 状态码 + 响应前 200 字符（截断）。
"""
import json
import sys
import urllib.request
from pathlib import Path

KEY_FILE = Path("/root/.damo_key")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.8-max"

key = KEY_FILE.read_text(encoding="utf-8").strip()
if not key:
    sys.exit("EMPTY_KEY_FILE")

body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": "回复两个字：收到"}],
    "max_tokens": 16,
    "temperature": 0.0,
}).encode("utf-8")

req = urllib.request.Request(
    f"{BASE_URL}/chat/completions", data=body, method="POST",
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
        print("HTTP", r.status)
        print("reply:", data["choices"][0]["message"]["content"][:100])
        print("model_returned:", data.get("model"))
        print("SMOKE_OK")
except urllib.error.HTTPError as e:
    print("HTTP_ERROR", e.code)
    print(e.read().decode("utf-8", errors="replace")[:300])  # 错误体不含 key
    sys.exit(1)
