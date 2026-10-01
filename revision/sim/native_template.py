"""Native chat-template sensitivity (D-TPL alternative, validation item 3c).

For a linear conversation, turn t's prompt is the tokenization of the full history
rendered by the tokenizer's own chat template up to user message t with the generation
prompt appended; its decode is the tokens the rendering of history-through-assistant-t
adds beyond that prompt. Unlike the fallback template, this is one rendering of the
whole history per turn, so (a) BOS appears once, (b) role markers follow the model,
(c) the tokenization of the history need not be prefix-stable across turns. Case (c)
is counted (`n_prefix_breaks`); when it happens the request array is rebuilt from the
new rendering (a real engine does the same), so causal reuse can be lost at that turn.
"""
from __future__ import annotations
from pathlib import Path
from .events import Request
from .ingest import IngestStats, MAX_TRANSCRIPT_TOKENS

ROLE_MAP = {"user": "user", "assistant": "assistant", "system": "system", "prompter": "user"}


class NativeTok:
    def __init__(self, name: str, tok_dir: Path):
        from transformers import AutoTokenizer
        d = Path(tok_dir) / name.replace("/", "__")
        self.tok = AutoTokenizer.from_pretrained(str(d))
        assert self.tok.chat_template, f"{name} ships no chat template"
        self.name = name

    def render_ids(self, msgs: list[dict], add_generation_prompt: bool) -> list[int]:
        chat = [{"role": ROLE_MAP.get(m["role"], m["role"]), "content": m["text"]} for m in msgs]
        return list(self.tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=add_generation_prompt))


def native_linear_group(rec: dict, ntok: NativeTok, stats: IngestStats) -> list[Request]:
    cid = rec["conv_id"]
    msgs = rec.get("messages") or []
    reqs: list[Request] = []
    prev_full: list[int] = []
    prev_rid = None
    prev_decode_path: tuple = ()
    t = 0
    i = 0
    # skip a leading assistant (not representable as a prompt under native templates)
    while i < len(msgs) and msgs[i]["role"] == "system":
        i += 1
    if i < len(msgs) and msgs[i]["role"] == "assistant":
        stats["n_groups_leading_assistant"] = stats.get("n_groups_leading_assistant", 0) + 1
        return []
    j = 0
    while j < len(msgs):
        # find next user message index j
        if msgs[j]["role"] != "user":
            j += 1
            continue
        prompt = ntok.render_ids(msgs[: j + 1], add_generation_prompt=True)
        if len(prompt) > MAX_TRANSCRIPT_TOKENS:
            stats["n_truncated_paths"] = stats.get("n_truncated_paths", 0) + 1
            break
        if prev_full and prompt[: len(prev_full)] != prev_full:
            stats["n_prefix_breaks"] = stats.get("n_prefix_breaks", 0) + 1
        dec: list[int] = []
        if j + 1 < len(msgs) and msgs[j + 1]["role"] == "assistant":
            full = ntok.render_ids(msgs[: j + 2], add_generation_prompt=False)
            if full[: len(prompt)] == prompt:
                dec = full[len(prompt):]
            else:
                stats["n_decode_prefix_breaks"] = stats.get("n_decode_prefix_breaks", 0) + 1
                dec = []
                full = prompt
            if len(full) > MAX_TRANSCRIPT_TOKENS:
                dec = dec[: MAX_TRANSCRIPT_TOKENS - len(prompt)]
                full = prompt + dec
            j += 2
        else:
            full = prompt
            j += 1
        rid = f"{cid}::{t}"
        path = prev_decode_path + (f"{cid}::u{t}",)
        decode_path = path + (f"{cid}::a{t}",)
        reqs.append(Request(rid, group=cid, path=path, prompt=prompt, decode=dec, parent=prev_rid, decode_path=decode_path))
        prev_full, prev_rid, prev_decode_path = full, rid, decode_path
        t += 1
    return reqs


def build_groups_native(records: list[dict], ntok: NativeTok):
    stats = IngestStats()
    groups = []
    seen: dict[str, int] = {}
    for rec in records:
        cid = str(rec["conv_id"]); k = seen.get(cid, 0); seen[cid] = k + 1
        if k:
            rec = dict(rec, conv_id=f"{cid}#{k + 1}")
        if (rec.get("kind") or "linear") != "linear":
            continue
        g = native_linear_group(rec, ntok, stats)
        if g:
            groups.append(g)
    stats["n_groups"] = len(groups); stats["n_requests"] = sum(len(g) for g in groups)
    return groups, stats
