from dataclasses import dataclass, field
MEDIUM_GPU = "GPU"
class KVCacheEvent: ...
@dataclass
class BlockStored(KVCacheEvent):
    block_hashes: list = field(default_factory=list)
    parent_block_hash: object = None
    token_ids: list = field(default_factory=list)
    block_size: int = 0
    lora_id: object = None
    medium: str = MEDIUM_GPU
    kv_cache_group_id: int = 0
    kv_cache_spec_kind: object = None
    sliding_window: object = None
@dataclass
class BlockRemoved(KVCacheEvent):
    block_hashes: list = field(default_factory=list)
    medium: str = MEDIUM_GPU
@dataclass
class AllBlocksCleared(KVCacheEvent):
    pass
