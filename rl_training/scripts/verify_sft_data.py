# -*- coding: utf-8 -*-
"""Independently verify the SFT-design reviewer's data-integrity claims (READ-ONLY).

Claims under test:
  C1. 433/1047 sft_train rows are label duplicates (~773 unique assistant labels, max 15 copies).
  C2. 23/50 syn_eval golden_patch match a training assistant label EXACTLY (normalized).
  C3. 29/50 syn_eval collide with train synthetic mutations on (file_rel, line_no); only 21/50 clean.
Prints structure dumps so field-name guesses can be corrected if a check returns a false 0.
"""
import json
import os
import re
import hashlib
from collections import Counter

D = "/root/autodl-tmp/sft_data"


def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        print(f"  !! missing {p}")
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def norm_patch(s):
    """Normalize a diff/patch for content equality: strip fences, CRLF, per-line trailing ws, blank edges."""
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


def assistant_text(rec):
    for m in rec.get("messages", []):
        if m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


def first_key(rec, *names):
    for n in names:
        if n in rec and rec[n] not in (None, ""):
            return rec[n]
    return ""


print("===== load =====")
train = load("sft_train.jsonl")
syn_eval = load("syn_eval.jsonl")
audit = load("sft_audit.jsonl")
print(f"  sft_train={len(train)}  syn_eval={len(syn_eval)}  sft_audit={len(audit)}")

print("\n===== structure dumps (first record keys) =====")
if train:
    print("  sft_train[0] top keys:", list(train[0].keys()))
    print("  sft_train[0] assistant label (first 240 chars):")
    print("    " + repr(assistant_text(train[0])[:240]))
if syn_eval:
    print("  syn_eval[0] keys:", list(syn_eval[0].keys()))
    gp = first_key(syn_eval[0], "golden_patch", "golden", "patch", "rollback_diff")
    print("  syn_eval[0] golden_patch (first 240 chars):")
    print("    " + repr((gp or "")[:240]))
if audit:
    print("  sft_audit[0] keys:", list(audit[0].keys()))

# ---------------- C1: label duplication in sft_train ----------------
print("\n===== C1: sft_train assistant-label duplication =====")
labels = [assistant_text(r) for r in train]
hashes = [hashlib.md5(l.encode("utf-8")).hexdigest() for l in labels]
cnt = Counter(hashes)
unique = len(cnt)
dup_groups = {k: v for k, v in cnt.items() if v > 1}
rows_in_dup = sum(dup_groups.values())
print(f"  total rows           : {len(train)}")
print(f"  unique labels        : {unique}")
print(f"  duplicate groups     : {len(dup_groups)}")
print(f"  rows in dup groups   : {rows_in_dup}  ({100*rows_in_dup//max(len(train),1)}%)")
print(f"  max copies of one    : {max(cnt.values()) if cnt else 0}")
print(f"  copy-count histogram : {dict(sorted(Counter(cnt.values()).items()))}")

# ---------------- C2: syn_eval exact-patch contamination ----------------
print("\n===== C2: syn_eval golden_patch EXACT match vs train labels =====")
train_norm_set = set(norm_patch(l) for l in labels if norm_patch(l))
exact_hits, hit_ids = 0, []
for i, ev in enumerate(syn_eval):
    gp = first_key(ev, "golden_patch", "golden", "patch", "rollback_diff")
    np_ = norm_patch(gp)
    if np_ and np_ in train_norm_set:
        exact_hits += 1
        hit_ids.append(first_key(ev, "syn_id", "id", "bug_id", "uid") or f"idx{i}")
print(f"  syn_eval records          : {len(syn_eval)}")
print(f"  EXACT match to a train lbl: {exact_hits}")
print(f"  clean (no exact match)    : {len(syn_eval) - exact_hits}")
print(f"  contaminated ids (<=30)   : {hit_ids[:30]}")

# ---------------- C3: (file_rel, line_no) collision ----------------
print("\n===== C3: syn_eval (file,line) collision vs train synthetic mutations =====")


def site_set(recs):
    out = set()
    for r in recs:
        f = first_key(r, "file_rel", "file", "rel_path", "path", "target_file")
        ln = first_key(r, "line_no", "line", "lineno")
        mut = first_key(r, "mutation", "mutation_type", "op")
        if f and ln != "":
            out.add((str(f), str(ln)))
        elif f and mut:
            out.add((str(f), str(mut)))
    return out


train_sites = site_set(audit) if audit else set()
ev_sites = site_set(syn_eval)
if not train_sites:
    print("  !! could not derive (file,line) sites from sft_audit — field names differ; dumping a synthetic audit row:")
    for r in audit:
        if str(first_key(r, "status", "audit", "kind")).lower() in ("pass", "skipped", "synthetic"):
            print("    ", json.dumps(r, ensure_ascii=False)[:300])
            break
coll = 0
coll_ids = []
for i, ev in enumerate(syn_eval):
    s = site_set([ev])
    if s and (s & train_sites):
        coll += 1
        coll_ids.append(first_key(ev, "syn_id", "id", "bug_id") or f"idx{i}")
print(f"  train synthetic sites : {len(train_sites)}")
print(f"  syn_eval sites        : {len(ev_sites)}")
print(f"  colliding records     : {coll}")
print(f"  colliding ids (<=30)  : {coll_ids[:30]}")

# ---------------- combined clean count ----------------
print("\n===== combined: truly-clean syn_eval (no exact match AND no site collision) =====")
clean = 0
for i, ev in enumerate(syn_eval):
    gp = first_key(ev, "golden_patch", "golden", "patch", "rollback_diff")
    s = site_set([ev])
    if (norm_patch(gp) not in train_norm_set) and not (s and (s & train_sites)):
        clean += 1
print(f"  clean records: {clean} / {len(syn_eval)}")
print("DONE")
