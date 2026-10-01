"""Regression guard for the sub-linear eviction structures: on random small
caches, the victim chosen by BlockCache must equal the brute-force choice over
the same candidate definition, for every policy and both leaf modes."""
import random

from sim.cache import BlockCache


def brute_victim(c: BlockCache, protected: set[bytes]):
    cands = [k for k, b in c.blocks.items()
             if b.pins == 0 and k not in protected and (not c.leaf_only or c.n_children(k) == 0)]
    if not cands:
        return None
    bs = [c.blocks[k] for k in cands]
    if c.policy == "LRU":
        return min(bs, key=lambda b: (b.last_access, b.insert_step, b.key)).key
    if c.policy == "LFU":
        return min(bs, key=lambda b: (b.count, b.insert_step, b.key)).key
    return min(bs, key=lambda b: (b.insert_step, b.key)).key


def test_victim_matches_brute_force_under_random_operations():
    rng = random.Random(3)
    for trial in range(60):
        policy = rng.choice(["LRU", "LFU", "FIFO"])
        leaf_only = rng.random() < 0.5
        c = BlockCache(capacity=10**9, policy=policy, B=4, leaf_only=leaf_only)
        keys = []
        step = 0
        for op in range(200):
            step += 1
            r = rng.random()
            if r < 0.45 or len(keys) < 3:
                k = bytes([rng.randrange(256) for _ in range(4)])
                if k in c.blocks:
                    continue
                parent = rng.choice(keys) if keys and rng.random() < 0.6 else None
                if parent is not None and parent not in c.blocks:
                    parent = None
                c.insert(k, parent, step, pinned=rng.random() < 0.2)
                keys.append(k)
            elif r < 0.65:
                k = rng.choice(keys)
                if k in c.blocks:
                    c.touch(k, step)
            elif r < 0.75:
                k = rng.choice(keys)
                if k in c.blocks and c.blocks[k].pins > 0:
                    c.unpin(k)
            else:
                protected = set(rng.sample(keys, min(len(keys), 2)))
                expect = brute_victim(c, protected)
                got = c.peek_victim(protected)
                assert got == expect, (trial, op, policy, leaf_only)
                if expect is not None:
                    assert c.evict_one(protected)
                    assert expect not in c.blocks


def test_reinserted_key_does_not_inherit_stale_heap_entry():
    """Fifth review: insert A, unpin, evict A, insert B, re-insert A; the next victim
    must be chosen from current metadata (B is older than the re-inserted A under
    every policy), not from a stale heap record of A's first life."""
    for policy in ("LRU", "LFU", "FIFO"):
        c = BlockCache(capacity=10**9, policy=policy, B=4, leaf_only=True)
        A, Bk = b"AAAA", b"BBBB"
        c.insert(A, None, step=1, pinned=True)
        c.unpin(A)                      # pushes an entry for A (version 1)
        assert c.evict_one(set())       # A gone; its entry may linger
        c.insert(Bk, None, step=2, pinned=False)
        c.insert(A, None, step=3, pinned=False)   # A's second life
        assert c.peek_victim(set()) == Bk, policy
        assert c.peek_victim(set()) == brute_victim(c, set()), policy


def test_victim_matches_brute_force_with_key_reuse():
    """Keys are evicted and re-inserted many times (small pool); stale heap entries
    from earlier lifetimes must never be taken as current."""
    rng = random.Random(11)
    n = 0
    for trial in range(400):
        policy = rng.choice(["LRU", "LFU", "FIFO"])
        leaf = rng.random() < 0.5
        c = BlockCache(capacity=10**9, policy=policy, B=4, leaf_only=leaf)
        pool = [bytes([i]) for i in range(6)]
        step = 0
        for _ in range(60):
            step += 1
            r = rng.random()
            k = rng.choice(pool)
            if r < 0.4:
                if k in c.blocks:
                    continue
                resident = [p for p in pool if p in c.blocks]
                parent = rng.choice(resident) if resident and rng.random() < 0.5 else None
                c.insert(k, parent, step, pinned=rng.random() < 0.3)
            elif r < 0.6:
                if k in c.blocks:
                    c.touch(k, step)
            elif r < 0.7:
                if k in c.blocks and c.blocks[k].pins > 0:
                    c.unpin(k)
            elif r < 0.8:
                if k in c.blocks and c.blocks[k].pins == 0:
                    c.pin(k)
            else:
                exp = brute_victim(c, set())
                assert c.peek_victim(set()) == exp, (trial, policy, leaf)
                if exp is not None:
                    c.evict_one(set())
                n += 1
    assert n > 2000
