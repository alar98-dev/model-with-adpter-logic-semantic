import os
import time
import threading
from collections import OrderedDict

from sensors.mmap_sensor import MmapSensoryEngine
from swa.workspace_policy import (
    get_watch_roots,
    should_skip_dir,
    should_index_file,
    _path_under_anomalous,
    MAX_TRACKED_FILES,
)


class DirectoryWatcherDaemon:
    def __init__(
        self,
        watch_path: str | None = None,
        watch_roots: list[str] | None = None,
        scan_interval: float = 12.0,
    ):
        if watch_roots is not None:
            self.watch_roots = watch_roots
        elif watch_path:
            self.watch_roots = [watch_path]
        else:
            self.watch_roots = get_watch_roots()

        self.scan_interval = scan_interval
        self.engine = MmapSensoryEngine(target_dim=640)
        self.active_spatial_map: OrderedDict = OrderedDict()
        self.phase_bridge = None
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
        self._pg_sync_boot_done = False
        self._verbose_pg = os.environ.get("SWA_WATCH_VERBOSE", "").strip().lower() in (
            "1", "true", "yes",
        )
        self._emfile_warned = False

    def attach_phase_bridge(self, bridge):
        self.phase_bridge = bridge
        if self._pg_sync_boot_done:
            return
        synced = 0
        with self._lock:
            pending = list(self.active_spatial_map.items())
        for filepath, info in pending:
            if not should_index_file(filepath):
                continue
            if self._sync_pgvector(filepath, info.get("sample", ""), info.get("vector"), quiet=True):
                synced += 1
        self._pg_sync_boot_done = True
        print(
            f"🌐 [PGVECTOR] Sync inicial: {synced} âncoras | "
            f"rastreados: {len(self.active_spatial_map)}/{MAX_TRACKED_FILES}"
        )

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True, name="SWA-DirWatcher")
        self._thread.start()
        zones = ", ".join(self.watch_roots[:5])
        if len(self.watch_roots) > 5:
            zones += f" (+{len(self.watch_roots) - 5})"
        print(f"📡 [WATCHER] Zonas: {zones} | cap={MAX_TRACKED_FILES} FD (sem mmap persistente)")

    def stop(self):
        self.running = False

    def get_spatial_map(self):
        with self._lock:
            return dict(self.active_spatial_map)

    def read_file_touch(self, filepath: str, max_bytes: int = 8000) -> str:
        """Toque psíquico sob demanda — não mantém arquivo aberto."""
        return MmapSensoryEngine.ler_conteudo(filepath, max_bytes=max_bytes)

    def _sync_pgvector(self, filepath: str, content_sample: str, vector_640, quiet: bool = False) -> bool:
        if self.phase_bridge is None or not should_index_file(filepath):
            return False
        try:
            interdim = self.phase_bridge.upsert_structural_anchor_from_mmap(
                filepath, content_sample, vector_640
            )
            if interdim and (self._verbose_pg or not quiet):
                print(f"🌐 [PGVECTOR] Âncora {interdim[:12]}… <- {filepath}")
            return bool(interdim)
        except Exception:
            return False

    def _evict_if_needed(self):
        while len(self.active_spatial_map) > MAX_TRACKED_FILES:
            self.active_spatial_map.popitem(last=False)

    def _scan_loop(self):
        self._perform_scan()
        while self.running:
            time.sleep(self.scan_interval)
            self._perform_scan()

    def _collect_files_under(self, root: str, max_per_root: int = 2000) -> list[str]:
        if not os.path.isdir(root) or _path_under_anomalous(root):
            return []

        found: list[str] = []
        try:
            for dirpath, dirnames, filenames in os.walk(root, topdown=True):
                if _path_under_anomalous(dirpath):
                    dirnames.clear()
                    continue
                dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]

                for file in filenames:
                    if len(found) >= max_per_root:
                        return found
                    filepath = os.path.join(dirpath, file)
                    if should_index_file(filepath):
                        try:
                            if os.path.getsize(filepath) > 200 * 1024:
                                continue
                        except OSError:
                            continue
                        found.append(filepath)
        except OSError as e:
            if e.errno == 24 and not self._emfile_warned:
                self._emfile_warned = True
                print("⚠️ [WATCHER] EMFILE ao varrer — reduzindo zonas; reinicie swa-auto se persistir.")
        except Exception as e:
            print(f"⚠️ [WATCHER] Erro em {root}: {e}")
        return found

    def _perform_scan(self):
        found_files: list[str] = []
        per_root = max(200, MAX_TRACKED_FILES // max(1, len(self.watch_roots)))
        for root in self.watch_roots:
            found_files.extend(self._collect_files_under(root, max_per_root=per_root))

        found_set = set(found_files)

        with self._lock:
            for filepath in found_files:
                try:
                    mtime = os.path.getmtime(filepath)
                    cached_info = self.active_spatial_map.get(filepath)

                    if cached_info is None or cached_info["mtime"] < mtime:
                        _mmap_unused, content_sample = self.engine.sentir_objeto(filepath)
                        if not content_sample and _mmap_unused is None:
                            continue
                        vector = self.engine.traduzir_para_vetor(filepath, content_sample)
                        self.active_spatial_map[filepath] = {
                            "vector": vector,
                            "mtime": mtime,
                            "sample": content_sample,
                        }
                        self.active_spatial_map.move_to_end(filepath)
                        self._evict_if_needed()
                        if self._verbose_pg:
                            print(f"🧲 [WATCHER] {filepath}")
                        self._sync_pgvector(filepath, content_sample, vector)
                except Exception:
                    continue

            for path in [p for p in self.active_spatial_map if p not in found_set]:
                del self.active_spatial_map[path]
