from dataclasses import dataclass, field
@dataclass
class PrefixCacheStats:
    reset: bool = False
    requests: int = 0
    queries: int = 0
    hits: int = 0
    preempted_requests: int = 0
    preempted_queries: int = 0
    preempted_hits: int = 0
    def record(self, num_tokens, num_hits, preempted=False):
        self.requests += 1; self.queries += num_tokens; self.hits += num_hits
        if preempted:
            self.preempted_requests += 1; self.preempted_queries += num_tokens; self.preempted_hits += num_hits
@dataclass
class PrefillStats: ...
@dataclass
class RequestSpecDecodeMetrics: ...
@dataclass
class KVCacheEvictionEvent:
    block_hash: object = None
    timestamp: float = 0.0
