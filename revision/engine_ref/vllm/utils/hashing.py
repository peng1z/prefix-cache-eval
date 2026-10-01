"""Stub: the comparison passes its own hash function to init_none_hash / the block
hasher; these names exist only so the vendored modules import."""
import hashlib, pickle
def sha256_pickle(obj) -> bytes:
    return hashlib.sha256(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).digest()
xxhash = sha256_pickle
xxhash_cbor = sha256_pickle
def get_hash_fn_by_name(name):
    return sha256_pickle
