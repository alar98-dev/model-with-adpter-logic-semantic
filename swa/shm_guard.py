import os
import tempfile

class ShmGuard:
    """[SWA BLINDAGEM] Cross-platform atomic file locking with system-native temp path."""
    def __init__(self, buffer_name="aeon_reality_buffer.tmp"):
        self.buffer_path = os.path.join(tempfile.gettempdir(), buffer_name)
        self.lock_file = f"{self.buffer_path}.lock"
        if not os.path.exists(self.lock_file):
            open(self.lock_file, 'a').close()
        self.fd = open(self.lock_file, 'r+')

    def acquire(self, exclusive=False):
        if os.name == 'nt':
            import msvcrt
            # msvcrt locking
            mode = 0 # LK_UNLCK
            if exclusive:
                mode = 2 # LK_LOCK
            else:
                mode = 2 # LK_LOCK
            msvcrt.locking(self.fd.fileno(), mode, 1)
        else:
            import fcntl
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(self.fd.fileno(), mode)

    def release(self):
        if os.name == 'nt':
            import msvcrt
            self.fd.seek(0)
            msvcrt.locking(self.fd.fileno(), 0, 1) # LK_UNLCK = 0
        else:
            import fcntl
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)

    def __enter__(self):
        self.acquire(exclusive=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
