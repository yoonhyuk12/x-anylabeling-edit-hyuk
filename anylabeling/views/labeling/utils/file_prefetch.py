"""Bounded read-ahead for images and annotation files on slow drives."""

import os
from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Thread

from ._io import io_path, open_file


@dataclass(frozen=True)
class _Snapshot:
    data: bytes
    signature: tuple | None


def _signature(path):
    stat = os.stat(io_path(path))
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


class FilePrefetchCache:
    """Cache immutable bytes, checking file identity before every reuse.

    Only daemon workers do speculative I/O. Replacing the read-ahead window
    drops queued work; clearing or closing also discards late results without
    waiting for a disconnected drive. Qt objects never enter these workers.
    """

    def __init__(self, max_bytes=128 * 1024 * 1024, workers=2):
        self.max_bytes = max_bytes
        self._worker_count = workers
        self._condition = Condition()
        self._cache = {}
        self._bytes = 0
        self._wanted = {}
        self._pending = deque()
        self._inflight = {}
        self._generation = 0
        self._started = False
        self._closed = False

    @staticmethod
    def _key(path):
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    @property
    def cached_bytes(self):
        with self._condition:
            return self._bytes

    def prefetch(self, paths):
        """Replace queued work with paths ordered from nearest to farthest."""
        wanted = dict.fromkeys(self._key(path) for path in paths)
        with self._condition:
            if self._closed:
                return
            self._wanted = {key: rank for rank, key in enumerate(wanted)}
            for key in list(self._cache):
                if key not in wanted:
                    self._remove(key)
            self._pending = deque(
                (key, self._generation)
                for key in wanted
                if key not in self._cache
                and (key, self._generation) not in self._inflight
            )
            if self._pending and not self._started:
                self._started = True
                for index in range(self._worker_count):
                    Thread(
                        target=self._worker,
                        daemon=True,
                        name=f"image-prefetch-{index}",
                    ).start()
            self._condition.notify_all()

    def read(self, path):
        """Read fresh bytes, sharing an in-flight read when one already exists."""
        key = self._key(path)
        while True:
            with self._condition:
                cached = self._cache.get(key)
                token = (key, self._generation)
                pending = self._inflight.get(token)
                if cached is None and pending is None:
                    pending = self._inflight[token] = Event()
                    break
            if cached is not None:
                try:
                    if cached.signature == _signature(key):
                        return cached.data
                except OSError:
                    with self._condition:
                        if self._cache.get(key) is cached:
                            self._remove(key)
                    raise
                with self._condition:
                    if self._cache.get(key) is cached:
                        self._remove(key)
            elif pending is not None:
                pending.wait()

        try:
            snapshot = self._read_snapshot(key)
            with self._condition:
                if token[1] == self._generation and not self._closed:
                    self._store(key, snapshot)
            return snapshot.data
        finally:
            with self._condition:
                self._inflight.pop(token, None)
                pending.set()
                self._condition.notify_all()

    def _read_snapshot(self, path, speculative=False):
        before = _signature(path)
        if speculative and before[2] > self.max_bytes:
            return None
        with open_file(path, "rb") as stream:
            data = stream.read(self.max_bytes + 1 if speculative else -1)
        after = _signature(path)
        # A concurrently changed file may be read normally, but never cached.
        return _Snapshot(data, after if before == after else None)

    def _remove(self, key):
        snapshot = self._cache.pop(key, None)
        if snapshot is not None:
            self._bytes -= len(snapshot.data)

    def _store(self, key, snapshot):
        if (
            snapshot is None
            or snapshot.signature is None
            or len(snapshot.data) > self.max_bytes
        ):
            return
        self._remove(key)
        self._cache[key] = snapshot
        self._bytes += len(snapshot.data)
        while self._bytes > self.max_bytes:
            # Preserve the next image when distant, large files fill the cache.
            farthest = max(
                self._cache,
                key=lambda name: self._wanted.get(name, len(self._wanted)),
            )
            self._remove(farthest)

    def _worker(self):
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                key, generation = token = self._pending.popleft()
                if token in self._inflight or key in self._cache:
                    self._condition.notify_all()
                    continue
                done = self._inflight[token] = Event()
            try:
                snapshot = self._read_snapshot(key, speculative=True)
                with self._condition:
                    if (
                        not self._closed
                        and generation == self._generation
                        and key in self._wanted
                    ):
                        self._store(key, snapshot)
            except Exception:
                # Missing sidecars and network errors are handled by the
                # normal foreground loader if the user opens that file.
                pass
            finally:
                with self._condition:
                    self._inflight.pop(token, None)
                    done.set()
                    self._condition.notify_all()

    def clear(self):
        with self._condition:
            self._generation += 1
            self._pending.clear()
            self._wanted.clear()
            self._cache.clear()
            self._bytes = 0

    def close(self):
        with self._condition:
            self._closed = True
            self.clear()
            self._condition.notify_all()
