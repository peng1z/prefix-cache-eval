"""Corpus jsonl -> groups of Requests under EVENT_MODEL v0.3 (R-S1..R-S5, R-T1..R-T3).

Reads the staged jsonl produced by the original provisioning (same records the
submitted run used) and applies the *new* request construction:
  linear : user message -> prefill, following assistant message -> decode
  tree   : regeneration reading (D-OASST default); path reading optional
  flat   : one request; decode from `response` if the record carries one
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .events import Request

MAX_TRANSCRIPT_TOKENS = 16384
SEED = 20260826


def hash01(s: str) -> float:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") / 2**64


def render(role: str, text: str) -> str:
    """Fallback template (R-T2, D-TPL default)."""
    return f"<|{role}|>\n{text}\n"


class Tok:
    def __init__(self, name: str, tok_dir: Path | None = None):
        self.name = name
        if name in ("cl100k_base", "o200k_base"):
            import tiktoken
            self._enc = tiktoken.get_encoding(name)
            self.encode = lambda s: self._enc.encode(s, disallowed_special=())
        else:
            from tokenizers import Tokenizer
            d = Path(tok_dir) / name.replace("/", "__")
            tj = next(d.rglob("tokenizer.json"))
            self._hf = Tokenizer.from_file(str(tj))
            self.encode = lambda s: self._hf.encode(s).ids



class IngestStats(dict):
    """Counters R-S2/R-S5 require: n_leading_assistant, n_truncated_paths, n_groups, n_requests."""


def canonical_bytes(rec: dict) -> bytes:
    """Same canonical form as the submitted pipeline: roles included, RS separators."""
    if rec.get("kind") == "tree":
        parts = [f"{n['role']}\n{n['text']}" for n in sorted(rec["nodes"], key=lambda x: x["id"])]
    else:
        parts = [f"{m['role']}\n{m['text']}" for m in rec.get("messages") or []]
    return "\n\x1e\n".join(parts).encode("utf-8")


def exact_dedup(records: list[dict]) -> tuple[list[dict], dict]:
    """Drop later records whose canonical bytes were already seen, independent of conv_id.
    Returns (kept, manifest)."""
    seen: set[str] = set()
    kept: list[dict] = []
    for r in records:
        h = hashlib.sha256(canonical_bytes(r)).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        kept.append(r)
    ids_blob = "\n".join(str(r["conv_id"]) for r in kept).encode()
    manifest = {
        "rule": "sha256(canonical_bytes: roles included, '\\n\\x1e\\n' separators), first occurrence kept, id-independent",
        "n_in": len(records), "n_out": len(kept), "n_dropped": len(records) - len(kept),
        "kept_ids_sha256": hashlib.sha256(ids_blob).hexdigest(),
    }
    return kept, manifest


def minhash_dedup(records: list[dict], nperm: int = 128, bands: int = 16, threshold: float = 0.9,
                  k: int = 5) -> tuple[list[dict], dict]:
    """MinHash/LSH near-duplicate stratum with the submitted pipeline's parameters
    (128 permutations, 16 bands, Jaccard 0.9, 5-word shingles over NFKC-lowercased canonical
    text), but keyed by record index rather than conv_id (FINDINGS_PHASE1 F1). Keeps the
    first record (in input order) of each cluster."""
    import re, unicodedata
    import numpy as np
    url_re = re.compile(r"https?://\S+")
    def normalize(t):
        t = unicodedata.normalize("NFKC", t); t = url_re.sub(" ", t); t = t.lower()
        return re.sub(r"\s+", " ", t).strip()
    rng = np.random.Generator(np.random.PCG64(SEED))
    a = rng.integers(1, 2**61 - 2, size=nperm, dtype=np.uint64)
    b = rng.integers(0, 2**61 - 2, size=nperm, dtype=np.uint64)
    MOD = np.uint64(2**61 - 1)
    sigs = []
    for r in records:
        words = normalize(canonical_bytes(r).decode("utf-8")).split()
        sh = [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)] if len(words) >= k else ([" ".join(words)] if words else [])
        if not sh:
            sigs.append(tuple([2**64 - 1] * nperm)); continue
        xs = np.array([int.from_bytes(hashlib.sha256(x.encode()).digest()[:8], "big") for x in sh], dtype=np.uint64)
        hv = (np.outer(xs, a) + b) % MOD
        sigs.append(tuple(int(v) for v in hv.min(axis=0)))
    rows = nperm // bands
    buckets: dict = defaultdict(list)
    for i, sig in enumerate(sigs):
        for band in range(bands):
            buckets[(band, sig[band * rows:(band + 1) * rows])].append(i)
    parent = list(range(len(records)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for members in buckets.values():
        if len(members) < 2: continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                si, sj = sigs[members[i]], sigs[members[j]]
                if sum(x == y for x, y in zip(si, sj)) / nperm >= threshold:
                    ri, rj = find(members[i]), find(members[j])
                    if ri != rj: parent[max(ri, rj)] = min(ri, rj)
    kept = [r for i, r in enumerate(records) if find(i) == i]
    return kept, {"rule": "minhash nperm=128 bands=16 jaccard>=0.9 5-word shingles, keyed by record",
                  "n_in": len(records), "n_out": len(kept), "n_dropped": len(records) - len(kept)}


def load_records(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: (str(r.get("timestamp") or ""), str(r["conv_id"])))
    return rows


def _truncate(ids: list[int]) -> tuple[list[int], bool]:
    if len(ids) > MAX_TRANSCRIPT_TOKENS:
        return ids[:MAX_TRANSCRIPT_TOKENS], True
    return ids, False


def linear_group(rec: dict, tok: Tok, stats: IngestStats) -> list[Request]:
    """R-S2. Returns the group's requests in order (one path)."""
    cid = rec["conv_id"]
    msgs = rec.get("messages") or []
    pieces = [tok.encode(render(m["role"], m["text"])) for m in msgs]
    reqs: list[Request] = []
    arr: list[int] = []
    prev_rid: str | None = None
    i = 0
    n = len(msgs)
    # leading system messages fold into the first prefill
    lead: list[int] = []
    while i < n and msgs[i]["role"] == "system":
        lead += pieces[i]
        i += 1
    arr += lead
    truncated = False
    t = 0
    first_non_system = i
    prev_decode_path: tuple[str, ...] = ()
    while i < n and not truncated:
        role = msgs[i]["role"]
        if role == "assistant":
            # assistant without a preceding user: empty-prefill request (R-S2, F6)
            stats["n_unpaired_assistant"] = stats.get("n_unpaired_assistant", 0) + 1
            if i == first_non_system:
                key = "n_unpaired_assistant_after_system" if lead else "n_groups_leading_assistant"
            else:
                key = "n_consecutive_assistant"
            stats[key] = stats.get(key, 0) + 1
            prompt = list(arr)
            dec = pieces[i]
            i += 1
        else:  # user (or system after a non-system message: treat as user-side text)
            arr += pieces[i]
            arr, truncated = _truncate(arr)
            prompt = list(arr)
            i += 1
            dec = []
            if not truncated and i < n and msgs[i]["role"] == "assistant":
                dec = pieces[i]
                i += 1
        full, trunc2 = _truncate(prompt + dec)
        if trunc2:
            dec = full[len(prompt):]
            truncated = True
        rid = f"{cid}::{t}"
        # path alternates user and assistant nodes so that turn t+1's path extends
        # turn t's decode_path (R-A1): (u0, a0, u1, a1, ..., u_t)
        path = prev_decode_path + (f"{cid}::u{t}",)
        decode_path = path + (f"{cid}::a{t}",)
        reqs.append(Request(rid, group=cid, path=path, prompt=prompt, decode=dec, parent=prev_rid,
                            decode_path=decode_path))
        arr = prompt + dec
        prev_rid = rid
        prev_decode_path = decode_path
        t += 1
    if truncated:
        stats["n_truncated_paths"] = stats.get("n_truncated_paths", 0) + 1
    return reqs


def tree_group(rec: dict, tok: Tok, stats: IngestStats, reading: str = "regeneration") -> list[Request]:
    """R-S3. regeneration reading: one group, DFS order, extra assistant children are
    regeneration requests. path reading: handled by `split_paths` on the same requests."""
    cid = rec["conv_id"]
    nodes = {n["id"]: n for n in rec["nodes"]}
    children: dict[str, list[str]] = defaultdict(list)
    roots = []
    for n in rec["nodes"]:
        p = n.get("parent")
        if p and p != "None" and p in nodes:
            children[p].append(n["id"])
        else:
            roots.append(n["id"])
    for p in children:
        children[p].sort()
    roots.sort()
    piece = {nid: tok.encode(render(n["role"], n["text"])) for nid, n in nodes.items()}
    reqs: list[Request] = []
    counter = [0]

    def new_rid() -> str:
        counter[0] += 1
        return f"{cid}::{counter[0]}"

    truncated_any = [False]

    def visit_user(nid: str, arr: list[int], path: tuple[str, ...], parent_rid: str | None):
        """A user node opens a request; each assistant child is its decode (first) or a
        regeneration (others); user children under each assistant child continue."""
        prompt, trunc = _truncate(arr + piece[nid])
        truncated_any[0] |= trunc
        path = path + (nid,)
        kids = children.get(nid, [])
        a_kids = [k for k in kids if nodes[k]["role"] == "assistant"]
        u_kids = [k for k in kids if nodes[k]["role"] != "assistant"]
        if trunc:
            rid = new_rid()
            reqs.append(Request(rid, group=cid, path=path, prompt=prompt, decode=[], parent=parent_rid))
            return
        if not a_kids:
            rid = new_rid()
            reqs.append(Request(rid, group=cid, path=path, prompt=prompt, decode=[], parent=parent_rid))
            for uk in u_kids:
                visit_user(uk, prompt, path, rid)
            return
        for j, ak in enumerate(a_kids):
            full, trunc2 = _truncate(prompt + piece[ak])
            truncated_any[0] |= trunc2
            dec = full[len(prompt):]
            rid = new_rid()
            reqs.append(Request(rid, group=cid, path=path, prompt=prompt, decode=dec,
                                parent=parent_rid, is_regeneration=(j > 0), decode_path=path + (ak,),
                                tree_sibling_rank=j))
            if trunc2:
                continue
            for uk in children.get(ak, []):
                if nodes[uk]["role"] == "assistant":
                    continue  # assistant under assistant: not a conversation shape we model
                visit_user(uk, full, path + (ak,), rid)
        for uk in u_kids:
            # user directly under user (rare): continuation without a reply
            visit_user(uk, prompt, path, reqs[-1].request_id)

    for r in roots:
        if nodes[r]["role"] == "assistant":
            stats["n_leading_assistant"] = stats.get("n_leading_assistant", 0) + 1
            continue
        visit_user(r, [], (), None)
    if truncated_any[0]:
        stats["n_truncated_paths"] = stats.get("n_truncated_paths", 0) + 1
    return reqs


def flat_group(rec: dict, tok: Tok, stats: IngestStats) -> list[Request]:
    """R-S4. prompt = rendered system + user; decode = response if present (D-ORCA)."""
    cid = rec["conv_id"]
    msgs = rec.get("messages") or []
    prompt: list[int] = []
    sys_len = 0
    for m in msgs:
        if m["role"] in ("system", "user"):
            piece = tok.encode(render(m["role"], m["text"]))
            if m["role"] == "system" and not prompt:
                sys_len = len(piece)
            prompt += piece
    prompt, trunc = _truncate(prompt)
    resp = rec.get("response")
    dec = tok.encode(render("assistant", resp)) if (resp and not trunc) else []
    full, trunc2 = _truncate(prompt + dec)
    dec = full[len(prompt):]
    if trunc or trunc2:
        stats["n_truncated_paths"] = stats.get("n_truncated_paths", 0) + 1
    if resp is None:
        stats["n_flat_without_response"] = stats.get("n_flat_without_response", 0) + 1
    return [Request(f"{cid}::0", group=cid, path=(f"{cid}::0",), prompt=prompt, decode=dec, parent=None,
                    decode_path=(f"{cid}::0", f"{cid}::a0"), sys_len=min(sys_len, len(prompt)))]


def split_paths(group: list[Request]) -> list[list[Request]]:
    """OASST path reading (D-OASST alternative): one group per root-to-leaf chain of requests."""
    by_id = {r.request_id: r for r in group}
    has_child = {r.parent for r in group if r.parent is not None}
    out: list[list[Request]] = []
    for leaf in [r for r in group if r.request_id not in has_child]:
        chain = []
        cur: Request | None = leaf
        while cur is not None:
            chain.append(cur)
            cur = by_id[cur.parent] if cur.parent is not None else None
        chain.reverse()
        gid = f"{leaf.group}/p{len(out)}"
        newid = {r.request_id: f"{r.request_id}@p{len(out)}" for r in chain}
        # Under the path reading each chain is an independent conversation: every request is the
        # first (and only) generation at its prompt point, so the execution-level regeneration flag
        # is False; the tree's branch identity survives in tree_sibling_rank (fourth review).
        out.append([Request(newid[r.request_id], group=gid, path=r.path, prompt=r.prompt, decode=r.decode,
                            parent=newid[r.parent] if r.parent is not None else None,
                            is_regeneration=False, decode_path=r.decode_path, sys_len=r.sys_len,
                            tree_sibling_rank=r.tree_sibling_rank)
                    for r in chain])
    return out


def legacy_linear_group(rec: dict, tok: Tok, stats: IngestStats) -> list[Request]:
    """Historical construction (submitted simulator): one prefill per non-leading-system
    message, assistant messages included; no decode events."""
    cid = rec["conv_id"]
    msgs = rec.get("messages") or []
    pieces = [tok.encode(render(m["role"], m["text"])) for m in msgs]
    arr: list[int] = []
    reqs: list[Request] = []
    seen_non_system = False
    prev = None
    t = 0
    for i, m in enumerate(msgs):
        arr = arr + pieces[i]
        arr, trunc = _truncate(arr)
        if m["role"] == "system" and not seen_non_system:
            continue
        seen_non_system = True
        rid = f"{cid}::{t}"
        path = tuple(f"{cid}::m{k}" for k in range(t + 1))
        reqs.append(Request(rid, group=cid, path=path, prompt=list(arr), decode=[], parent=prev))
        prev = rid
        t += 1
        if trunc:
            stats["n_truncated_paths"] = stats.get("n_truncated_paths", 0) + 1
            break
    return reqs


def legacy_tree_group(rec: dict, tok: Tok, stats: IngestStats) -> list[Request]:
    """Historical construction for trees: one prefill per node (the old `walk`), DFS order."""
    cid = rec["conv_id"]
    nodes = {n["id"]: n for n in rec["nodes"]}
    children: dict[str, list[str]] = defaultdict(list)
    roots = []
    for n in rec["nodes"]:
        p = n.get("parent")
        if p and p != "None" and p in nodes:
            children[p].append(n["id"])
        else:
            roots.append(n["id"])
    for p in children:
        children[p].sort()
    roots.sort()
    piece = {nid: tok.encode(render(n["role"], n["text"])) for nid, n in nodes.items()}
    reqs: list[Request] = []

    def walk(nid, arr, path, parent_rid):
        prompt, trunc = _truncate(arr + piece[nid])
        rid = f"{cid}::{nid}"
        reqs.append(Request(rid, group=cid, path=path + (nid,), prompt=prompt, decode=[], parent=parent_rid))
        if trunc:
            return
        for ch in children.get(nid, []):
            walk(ch, prompt, path + (nid,), rid)

    for r in roots:
        walk(r, [], (), None)
    return reqs


def build_groups(records: list[dict], tok: Tok, stats: IngestStats | None = None,
                 tree_reading: str = "regeneration", leading_assistant: str = "decode",
                 legacy_assistant_prefill: bool = False) -> tuple[list[list[Request]], IngestStats]:
    """tree_reading: 'regeneration' | 'path' (D-OASST). leading_assistant: 'decode' (empty-prefill
    request whose decode is the message) | 'exclude' (drop groups whose first non-system message is
    an assistant message; consecutive assistant messages mid-conversation are kept) (F6)."""
    if tree_reading not in ("regeneration", "path"):
        raise ValueError(tree_reading)
    if leading_assistant not in ("decode", "exclude"):
        raise ValueError(leading_assistant)
    stats = stats if stats is not None else IngestStats()
    groups: list[list[Request]] = []
    seen_ids: dict[str, int] = {}
    for rec in records:
        # R-S1: group ids must be unique. The staged WildChat frame repeats some
        # conv_ids (see FINDINGS_PHASE1 F1); later copies get a "#k" suffix.
        cid = str(rec["conv_id"])
        k = seen_ids.get(cid, 0)
        seen_ids[cid] = k + 1
        if k:
            stats["n_duplicate_conv_ids"] = stats.get("n_duplicate_conv_ids", 0) + 1
            rec = dict(rec, conv_id=f"{cid}#{k + 1}")
        kind = rec.get("kind") or "linear"
        if legacy_assistant_prefill and kind in ("tree", "linear"):
            g = legacy_tree_group(rec, tok, stats) if kind == "tree" else legacy_linear_group(rec, tok, stats)
            if g:
                groups.append(g)
            continue
        if kind == "tree":
            g = tree_group(rec, tok, stats, tree_reading)
            if tree_reading == "path":
                for pg in split_paths(g):
                    groups.append(pg)
                continue
        elif kind == "flat":
            g = flat_group(rec, tok, stats)
        else:
            before = stats.get("n_groups_leading_assistant", 0) + stats.get("n_unpaired_assistant_after_system", 0)
            g = linear_group(rec, tok, stats)
            after = stats.get("n_groups_leading_assistant", 0) + stats.get("n_unpaired_assistant_after_system", 0)
            if leading_assistant == "exclude" and after > before:
                stats["n_groups_excluded_leading_assistant"] = stats.get("n_groups_excluded_leading_assistant", 0) + 1
                continue
        if g:
            groups.append(g)
    stats["n_groups"] = len(groups)
    stats["n_requests"] = sum(len(g) for g in groups)
    return groups, stats


def subsample(records: list[dict], n_target: int, salt: str) -> list[dict]:
    """Seeded conversation subsample independent of ingest sampling (uses a fresh salt)."""
    if len(records) <= n_target:
        return records
    thresh = n_target / len(records)
    return [r for r in records if hash01(f"{r['conv_id']}:{salt}:{SEED}") < thresh]
