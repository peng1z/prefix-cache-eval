#!/usr/bin/env python3
"""Download public corpora and emit compact conversation JSONL.

Sampling (preregistered, idea_spec §5 / blind-spot #6):
  keep iff int.from_bytes(sha256(f"{conv_id}:{SEED}")[:8]) / 2**64 < rate
  rates: wildchat 0.05, openorca 0.05, oasst 1.0, sharegpt 1.0
  whole conversations/rows only; never turn-level truncation.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

SEED = 20260826
PAY = Path(__file__).resolve().parent
RAW = PAY / "raw"
DER = PAY / "derived"
TOK = PAY / "tokenizers"
RAW.mkdir(parents=True, exist_ok=True)
DER.mkdir(parents=True, exist_ok=True)
TOK.mkdir(parents=True, exist_ok=True)

RATES = {
    "wildchat": 0.05,
    "openorca": 0.05,
    "oasst": 1.0,
    "sharegpt": 1.0,
}


def keep_id(cid: str, rate: float) -> bool:
    if rate >= 1.0:
        return True
    h = hashlib.sha256(f"{cid}:{SEED}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64) < rate


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dl(repo: str, filename: str, dest_dir: Path, repo_type: str = "dataset") -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"DOWNLOAD {repo} {filename}", flush=True)
    t0 = time.time()
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type=repo_type,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    p = Path(path)
    print(f"  -> {p} {p.stat().st_size} bytes in {time.time()-t0:.1f}s", flush=True)
    return p


def ingest_sharegpt() -> dict:
    p = dl(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "ShareGPT_V3_unfiltered_cleaned_split.json",
        RAW / "sharegpt",
    )
    out = DER / "sharegpt.jsonl"
    n_raw = n_keep = n_drop_empty = 0
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(out, "w", encoding="utf-8") as w:
        for i, row in enumerate(data):
            n_raw += 1
            cid = str(row.get("id") or f"sg-{i}")
            if not keep_id(cid, RATES["sharegpt"]):
                continue
            conv = row.get("conversations") or []
            msgs = []
            for t in conv:
                role_raw = (t.get("from") or "").strip().lower()
                text = t.get("value") or t.get("text") or ""
                if not isinstance(text, str) or not text.strip():
                    continue
                if role_raw in ("human", "user", "prompter"):
                    role = "user"
                elif role_raw in ("gpt", "assistant", "chatgpt", "bing"):
                    role = "assistant"
                elif role_raw in ("system",):
                    role = "system"
                else:
                    role = "user" if len(msgs) % 2 == 0 else "assistant"
                msgs.append({"role": role, "text": text.strip()})
            if not msgs:
                n_drop_empty += 1
                continue
            n_keep += 1
            w.write(
                json.dumps(
                    {
                        "corpus": "sharegpt",
                        "conv_id": cid,
                        "timestamp": None,
                        "kind": "linear",
                        "messages": msgs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "corpus": "sharegpt",
        "file": str(p),
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
        "n_raw": n_raw,
        "n_keep": n_keep,
        "n_drop_empty": n_drop_empty,
        "sample_rate": RATES["sharegpt"],
        "out": str(out),
    }


def ingest_oasst() -> dict:
    p = dl(
        "OpenAssistant/oasst1",
        "data/train-00000-of-00001-b42a775f407cee45.parquet",
        RAW / "oasst",
    )
    import pyarrow.parquet as pq

    table = pq.read_table(p)
    cols = {c: table.column(c).to_pylist() for c in table.column_names}
    n = len(next(iter(cols.values())))
    trees: dict[str, list] = {}
    n_skip_deleted = 0
    for i in range(n):
        deleted = cols.get("deleted", [False] * n)[i]
        if deleted:
            n_skip_deleted += 1
            continue
        tree_id = str(cols["message_tree_id"][i])
        trees.setdefault(tree_id, []).append(
            {
                "id": str(cols["message_id"][i]),
                "parent": None
                if cols["parent_id"][i] is None
                else str(cols["parent_id"][i]),
                "role": str(cols["role"][i] or "prompter"),
                "text": str(cols["text"][i] or "").strip(),
                "created": str(cols["created_date"][i] or ""),
                "tree_state": str(cols.get("tree_state", [""] * n)[i] or ""),
            }
        )
    out = DER / "oasst.jsonl"
    n_keep = n_empty = 0
    with open(out, "w", encoding="utf-8") as w:
        for tree_id, nodes in trees.items():
            if not keep_id(tree_id, RATES["oasst"]):
                continue
            nodes = [nd for nd in nodes if nd["text"]]
            if not nodes:
                n_empty += 1
                continue
            ts = min((nd["created"] for nd in nodes if nd["created"]), default=None)
            n_keep += 1
            w.write(
                json.dumps(
                    {
                        "corpus": "oasst",
                        "conv_id": tree_id,
                        "timestamp": ts,
                        "kind": "tree",
                        "nodes": [
                            {
                                "id": nd["id"],
                                "parent": nd["parent"],
                                "role": "user"
                                if nd["role"] in ("prompter", "user")
                                else "assistant",
                                "text": nd["text"],
                            }
                            for nd in nodes
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "corpus": "oasst",
        "file": str(p),
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
        "n_raw_messages": n,
        "n_skip_deleted": n_skip_deleted,
        "n_trees_raw": len(trees),
        "n_keep": n_keep,
        "n_empty": n_empty,
        "sample_rate": RATES["oasst"],
        "out": str(out),
        "split": "HF default train parquet (ready_for_export messages)",
    }


def ingest_openorca() -> dict:
    p = dl(
        "Open-Orca/OpenOrca",
        "1M-GPT4-Augmented.parquet",
        RAW / "openorca",
    )
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(p)
    out = DER / "openorca.jsonl"
    n_raw = n_keep = n_empty = 0
    with open(out, "w", encoding="utf-8") as w:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            names = table.column_names
            cols = {c: table.column(c).to_pylist() for c in names}
            nrow = len(next(iter(cols.values())))
            id_col = "id" if "id" in cols else None
            sys_col = "system_prompt" if "system_prompt" in cols else "system"
            q_col = "question" if "question" in cols else "instruction"
            for i in range(nrow):
                n_raw += 1
                cid = str(cols[id_col][i]) if id_col else f"orca-{n_raw}"
                if not keep_id(cid, RATES["openorca"]):
                    continue
                sys_t = cols.get(sys_col, [""] * nrow)[i] or ""
                q_t = cols.get(q_col, [""] * nrow)[i] or ""
                if not isinstance(sys_t, str):
                    sys_t = str(sys_t)
                if not isinstance(q_t, str):
                    q_t = str(q_t)
                sys_t, q_t = sys_t.strip(), q_t.strip()
                if not q_t:
                    n_empty += 1
                    continue
                msgs = []
                if sys_t:
                    msgs.append({"role": "system", "text": sys_t})
                msgs.append({"role": "user", "text": q_t})
                n_keep += 1
                w.write(
                    json.dumps(
                        {
                            "corpus": "openorca",
                            "conv_id": cid,
                            "timestamp": None,
                            "kind": "flat",
                            "messages": msgs,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return {
        "corpus": "openorca",
        "file": str(p),
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
        "n_raw": n_raw,
        "n_keep": n_keep,
        "n_empty": n_empty,
        "sample_rate": RATES["openorca"],
        "out": str(out),
        "split": "1M-GPT4-Augmented.parquet (GPT-4 FLAN-augmented; not the 3.5M GPT-3.5 file)",
    }


def ingest_wildchat() -> dict:
    shards = [f"data/train-{i:05d}-of-00014.parquet" for i in range(14)]
    out = DER / "wildchat.jsonl"
    n_raw = n_keep = n_empty = 0
    shard_meta = []
    with open(out, "w", encoding="utf-8") as w:
        for fn in shards:
            p = dl("allenai/WildChat-1M", fn, RAW / "wildchat")
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(p)
            n_shard = 0
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                cols = {c: table.column(c).to_pylist() for c in table.column_names}
                nrow = len(next(iter(cols.values())))
                ch = cols.get("conversation_hash") or cols.get("hash")
                convs = cols.get("conversation")
                ts = cols.get("timestamp") or cols.get("date")
                for i in range(nrow):
                    n_raw += 1
                    n_shard += 1
                    cid = str(ch[i]) if ch else f"wc-{n_raw}"
                    if not keep_id(cid, RATES["wildchat"]):
                        continue
                    conv = convs[i] if convs else None
                    msgs = []
                    if isinstance(conv, list):
                        for t in conv:
                            if not isinstance(t, dict):
                                continue
                            role = str(t.get("role") or "user").lower()
                            text = t.get("content") or t.get("text") or ""
                            if not isinstance(text, str) or not text.strip():
                                continue
                            if role not in ("user", "assistant", "system"):
                                role = "user"
                            msgs.append({"role": role, "text": text.strip()})
                    if not msgs:
                        n_empty += 1
                        continue
                    n_keep += 1
                    stamp = None
                    if ts:
                        stamp = str(ts[i]) if ts[i] is not None else None
                    w.write(
                        json.dumps(
                            {
                                "corpus": "wildchat",
                                "conv_id": cid,
                                "timestamp": stamp,
                                "kind": "linear",
                                "messages": msgs,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            shard_meta.append(
                {
                    "file": fn,
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                    "n_rows": n_shard,
                }
            )
            # drop shard after ingest to cap disk; hashes already recorded
            try:
                p.unlink()
            except OSError:
                pass
    return {
        "corpus": "wildchat",
        "n_raw": n_raw,
        "n_keep": n_keep,
        "n_empty": n_empty,
        "sample_rate": RATES["wildchat"],
        "out": str(out),
        "shards": shard_meta,
        "note": "shards hashed then deleted after ingest; sample is whole conversations by seeded hash",
    }


def fetch_tokenizers() -> list:
    specs = [
        ("Qwen/Qwen2.5-0.5B", ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", ["tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json"]),
        ("microsoft/Phi-3-mini-4k-instruct", ["tokenizer.json", "tokenizer_config.json", "tokenizer.model"]),
        ("HuggingFaceTB/SmolLM2-135M", ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]),
    ]
    meta = []
    for repo, files in specs:
        dest = TOK / repo.replace("/", "__")
        got = []
        for fn in files:
            try:
                p = dl(repo, fn, dest, repo_type="model")
                got.append({"file": fn, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
            except Exception as e:
                got.append({"file": fn, "error": str(e)})
        meta.append({"repo": repo, "files": got, "gated": False})
    return meta


def main():
    t0 = time.time()
    reports = {}
    reports["sharegpt"] = ingest_sharegpt()
    reports["oasst"] = ingest_oasst()
    reports["openorca"] = ingest_openorca()
    reports["wildchat"] = ingest_wildchat()
    reports["tokenizers"] = fetch_tokenizers()
    reports["seed"] = SEED
    reports["rates"] = RATES
    reports["wall_s"] = round(time.time() - t0, 1)
    (DER / "ingest_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print("INGEST_DONE", json.dumps({k: {kk: vv for kk, vv in (v.items() if isinstance(v, dict) else []) if kk != "shards"} if isinstance(v, dict) else v for k, v in reports.items() if k != "tokenizers"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
