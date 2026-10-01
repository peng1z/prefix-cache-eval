class _Platform:
    def is_cuda(self): return False
    def is_rocm(self): return False
    def is_cpu(self): return True
    def is_tpu(self): return False
    def is_xpu(self): return False
    def is_cuda_alike(self): return False
    device_type = "cpu"
    def __getattr__(self, name):
        # any other capability query: answer "no" (returns a callable that returns False)
        return lambda *a, **k: False
current_platform = _Platform()
