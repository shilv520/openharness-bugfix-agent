# -*- coding: utf-8 -*-
"""Filter syn_eval.jsonl -> syn_eval_clean.jsonl keeping only records with
   (a) NO exact normalized golden_patch match against any sft_train assistant label, AND
   (b) NO (file_rel, line_no) collision against sft_audit synthetic mutation sites.
   Conservative clean gate. Read sft_data, write one new file. READ-ONLY otherwise."""
import json
import os
import re
from collections import Counter

D = "/root/autodl-tmp/sft_data"


def load(n):
    p = os.path.join(D, n)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def norm_patch(s):
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"^```[a-zA-Z]*\n", "", s)
    s = re.sub(r"\n```$", "", s)
    lines = [ln.rstrip() for ln in s.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def atext(r):
    for m in r.get("messages", []):
        if m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


train = load("sft_train.jsonl")
syn = load("syn_eval.jsonl")
audit = load("sft_audit.jsonl")

train_norm = set(norm_patch(atext(r)) for r in train)
train_norm.discard("")


def site(r, fkey, lkey):
    f = r.get(fkey) or ""
    ln = r.get(lkey)
    return (str(f), str(ln)) if f and ln not in (None, "") else None


train_sites = set(s for s in (site(r, "file", "line_no") for r in audit) if s)

clean = []
for ev in syn:
    gp = ev.get("golden_patch", "") or ""
    if norm_patch(gp) in train_norm:
        continue
    s = site(ev, "file_rel", "line_no")
    if s and s in train_sites:
        continue
    clean.append(ev)

out = os.path.join(D, "syn_eval_clean.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for ev in clean:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")

print(f"clean: {len(clean)}/{len(syn)} -> {out}")


def srcbug(ev):
    bid = ev.get("bug_id", "")
    m = re.match(r"SYN-([A-Za-z]+-\d+)-", str(bid))
    return m.group(1) if m else str(bid)


print("clean per source bug:", dict(Counter(srcbug(ev) for ev in clean)))
print("clean mutations:", dict(Counter(ev.get("mutation", "?") for ev in clean)))
print("DONE")
