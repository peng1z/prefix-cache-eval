"""Requests, events and chain-hashed block keys (EVENT_MODEL R-S1, R-B1, R-E1, R-E2)."""
from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass, field

ZERO_PARENT = bytes(32)


@dataclass(frozen=True)
class Request:
    """One generation request (R-S1): a prefill of `prompt` followed by a decode of `decode`.

    prompt: full token array prefix the prefill sees (system + history + user).
    decode: tokens appended by the assistant reply (may be empty).
    parent: request_id of the request whose decode produced this request's
            parent assistant node, or None (root of its group).
    """

    request_id: str
    group: str
    path: tuple[str, ...]
    prompt: list[int]
    decode: list[int]
    parent: str | None
    is_regeneration: bool = False
    decode_path: tuple[str, ...] = ()   # path of the assistant node this decode produces (R-A1)
    sys_len: int = 0                     # per-row system span for flat corpora (R-A3)
    tree_sibling_rank: int = 0           # source field: index of this assistant child in its tree (0 = first)


@dataclass
class Event:
    kind: str  # "P" prefill, "D" decode
    request: Request


_KEY_MEMO: dict[tuple[bytes, bytes], bytes] = {}


def chain_keys(ids: list[int], B: int) -> list[bytes]:
    """Chain keys of every full block of `ids` (R-B1, R-B2).

    Memoised on (parent_key, block_bytes): requests that share a prefix reuse the
    same entries, so keying a prefix-heavy corpus costs one SHA-256 per distinct
    (parent, block) pair rather than one per request per block.
    """
    keys: list[bytes] = []
    parent = ZERO_PARENT
    for i in range(len(ids) // B):
        block = array("i", ids[i * B : (i + 1) * B]).tobytes()
        mk = (parent, block)
        h = _KEY_MEMO.get(mk)
        if h is None:
            h = hashlib.sha256(parent + block).digest()
            _KEY_MEMO[mk] = h
        keys.append(h)
        parent = h
    return keys


def chain_key_one(parent: bytes | None, block_ids) -> bytes:
    """Key of one full block given its parent key (ZERO_PARENT for block 0)."""
    parent = ZERO_PARENT if parent is None else parent
    block = array("i", block_ids).tobytes()
    mk = (parent, block)
    h = _KEY_MEMO.get(mk)
    if h is None:
        h = hashlib.sha256(parent + block).digest()
        _KEY_MEMO[mk] = h
    return h


def clear_key_memo() -> None:
    _KEY_MEMO.clear()


def prompt_keys(r: Request, B: int) -> list[bytes]:
    """Keys of the full blocks a prefill demands."""
    return chain_keys(r.prompt, B)


def decode_keys(r: Request, B: int) -> list[bytes]:
    """Keys of the blocks a decode completes: those whose end lies beyond the prompt (R-E2)."""
    full = chain_keys(r.prompt + r.decode, B)
    n_prompt_blocks = len(r.prompt) // B
    return full[n_prompt_blocks:]
