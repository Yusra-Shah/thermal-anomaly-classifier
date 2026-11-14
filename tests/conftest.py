"""pytest hooks: report the peak memory of the process after the session (Windows)."""


def pytest_sessionfinish(session, exitstatus):
    try:
        from tests.test_live import peak_memory
        ws, commit = peak_memory()
        print(f"\npeak memory: working set {ws:.0f} MB, commit {commit:.0f} MB")
    except Exception:
        pass
