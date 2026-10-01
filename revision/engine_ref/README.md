# Serial vLLM cache component comparison

## What is real vLLM code

Vendored unchanged from `vllm-project/vllm` at commit
`8f816a3f665489d7f0d222115d4f72ebab01076b` under `engine_ref/vllm/`:

| file | role |
|---|---|
| `v1/core/kv_cache_manager.py` | `KVCacheManager`: `get_computed_blocks`, `allocate_slots`, `free`, `cache_blocks` |
| `v1/core/kv_cache_coordinator.py` | unitary coordinator over one KV-cache group |
| `v1/core/single_type_kv_cache_manager.py` | `FullAttentionManager`: longest-prefix hit, allocation, caching |
| `v1/core/block_pool.py` | `BlockPool`, cached-hash map, eviction from the free queue |
| `v1/core/kv_cache_utils.py` | `KVCacheBlock`, `FreeKVCacheBlockQueue`, block hashing (`get_request_block_hasher`, `hash_block_tokens`) |
| `v1/kv_cache_interface.py`, `v1/kv_cache_spec_registry.py`, `v1/kv_cache_layout.py`, `v1/core/kv_cache_metrics.py` | spec/config types used by the above |

## What is ours (stubs)

Minimal replacements so the above import without the rest of vLLM:
`vllm/__init__.py`, `envs.py`, `logger.py`, `config/__init__.py`
(`VllmConfig` placeholder, `get_current_vllm_config_or_none → None`),
`utils/*` (`cdiv`, `round_up`, `round_down`, `format_gib`, `get_dtype_size`,
`length_from_prompt_token_ids_or_embeds`), `utils/hashing.py` (names only;
the comparison passes its own SHA-256-over-pickle hash function),
`distributed/kv_events.py` (event dataclasses), `v1/metrics/stats.py`,
`v1/utils.py`, `v1/attention/backends/registry.py`, `platforms/__init__.py`
(a CPU platform answering "no" to every capability), and `v1/request.py`, a
minimal `Request` carrying only the attributes the core reads
(`prompt_token_ids`, `all_token_ids`, `num_tokens`, `num_computed_tokens`,
`block_hashes` with vLLM's `update_block_hashes` contract, and `None`/empty
for LoRA, multimodal, prompt embeds, cache salt, session id).

`torch` (CPU build) is imported by `kv_cache_interface` for dtype sizes only.

## Harness (`harness.py`)

What this is: a **component comparison** of the vendored cache core under
a driver we wrote, not a validation against a running vLLM engine. Ours are
the `Request` object (vLLM's `v1/request.py` is replaced, not vendored),
the hash function (SHA-256 over pickle in place of `utils/hashing.py`), the
serial loop, and the completion-time `cache_blocks` call. The scheduler is
not vendored. Only per-request hit counts are compared; no eviction trace
is recorded, so victim identity is not compared.

Serial replay, one request at a time, of the simulator's event-model requests:
`prefill` = `get_computed_blocks` + `allocate_slots(num_new_tokens = L − hit,
num_new_computed_tokens = hit)`; decode in block-sized steps with the real
order `allocate_slots(n)` then `append_output_token_ids`; completion =
optional `cache_blocks(request, num_computed_tokens)` then `free`.

The `cache_blocks` call at completion is a harness choice: in vLLM the blocks
completed by the last decode tokens enter the cache map at the *next*
scheduler step, which a finished request never has. Both choices are run and
reported (`cache_on_complete` True/False); the difference between them is
a difference between two driver conventions, and whether the real
scheduler leaves those blocks unhashed for a finished request is not
checked here.

## Semantics that differ from the simulator by construction

| aspect | vLLM core | simulator (main model) |
|---|---|---|
| eviction order | LRU over *release time*: a block enters the free queue when its last user frees it; blocks of one request are freed tail-first | LRU over last access; leaf-only candidates |
| candidates | any released block; an evicted ancestor leaves descendants unmatchable (longest-prefix scan stops) | leaf-only (ablated: `any_unpinned`) |
| policies | LRU only | LRU, LFU, FIFO |
| full-hit rule | hit capped at `num_tokens − 1` | R-E4 correction (on) |
| admission | optimistic: allocate if free blocks suffice now; preemption exists in the scheduler (not exercised here) | full-footprint reservation, no preemption |
| concurrency | this harness is serial only | `K_exec` slots |

The synthetic tests exercise infinite capacity reuse and the full hit rule.
Finite capacity comparisons are sensitive to eviction ordering and candidate
eligibility. The reference driver supports serial component evaluation;
concurrent engine scheduling and timing require a separate evaluation.

## Known non-representable case

vLLM rejects an empty prompt. The simulator's D-LEAD `decode` condition
creates empty-prefill requests whose decode is written to the cache (a
conversation whose first message is an assistant reply). Such requests are
skipped by the harness, so their decode blocks are absent on the engine side
and the next prefill of that group misses where the simulator hits. The
comparison therefore runs under the `exclude` condition (those groups
dropped), and the synthetic test drops groups containing an empty prompt.
This is a limitation of representing chunked transcripts, not a cache-logic
difference.

## Running the comparison

From the repository root:

```sh
PYTHONPATH=revision python revision/engine_ref/run_compare.py --data-dir work/data/derived --out work/runs/component
```

Synthetic comparisons are part of `revision/sim/tests/test_engine_ref.py`.
The runner writes fresh comparison outputs to the selected directory.
