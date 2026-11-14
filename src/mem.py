"""Peak memory of this process (Windows): peak working set and peak commit in MB."""


def peak_memory():
    import ctypes
    from ctypes import wintypes as w

    class PMC(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("faults", w.DWORD), ("peak_ws", ctypes.c_size_t),
                    ("ws", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                    ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("commit", ctypes.c_size_t),
                    ("peak_commit", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(c)
    k, p = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k.GetCurrentProcess.restype = w.HANDLE
    p.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD]
    p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.peak_ws / 2 ** 20, c.peak_commit / 2 ** 20


def report(stage):
    ws, commit = peak_memory()
    print(f"[memory] {stage}: peak working set {ws:.0f} MB, peak commit {commit:.0f} MB", flush=True)
    return commit
