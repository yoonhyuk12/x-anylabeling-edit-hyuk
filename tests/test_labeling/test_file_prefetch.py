import json
import os
import threading
from unittest import mock

import pytest
from PIL import Image

from anylabeling.views.labeling.label_file import LabelFile
from anylabeling.views.labeling.utils import file_prefetch
from anylabeling.views.labeling.utils.file_prefetch import FilePrefetchCache


@pytest.fixture
def cache():
    instance = FilePrefetchCache()
    yield instance
    instance.close()


def wait_idle(cache):
    with cache._condition:
        assert cache._condition.wait_for(
            lambda: not cache._pending and not cache._inflight, timeout=5
        ), "background reads did not finish"


def test_prefetch_reads_four_files_concurrently_and_reuses_bytes(
    cache, tmp_path, monkeypatch
):
    files = [tmp_path / f"image-{i}.jpg" for i in range(5)]
    for path in files:
        path.write_bytes(path.name.encode())
    both_started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    readers = []
    real_open = file_prefetch.open_file

    def slow_open(*args, **kwargs):
        with lock:
            readers.append(threading.get_ident())
            if len(readers) == 4:
                both_started.set()
        assert release.wait(5)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(file_prefetch, "open_file", slow_open)
    cache.prefetch(files)
    try:
        assert both_started.wait(3)
        assert threading.get_ident() not in readers
        assert len(readers) == 4  # The fifth read waits for a worker.
    finally:
        release.set()
    wait_idle(cache)
    with mock.patch.object(
        file_prefetch, "open_file", side_effect=AssertionError("reread")
    ):
        for path in files:
            assert cache.read(path) == path.name.encode()


def test_foreground_shares_an_inflight_read(cache, tmp_path, monkeypatch):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"image")
    started, release = threading.Event(), threading.Event()
    real_open = file_prefetch.open_file
    opens = []

    def slow_open(*args, **kwargs):
        opens.append(args[0])
        started.set()
        assert release.wait(5)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(file_prefetch, "open_file", slow_open)
    cache.prefetch([path])
    results = []
    reader = threading.Thread(target=lambda: results.append(cache.read(path)))
    try:
        assert started.wait(3)
        reader.start()
    finally:
        release.set()
    reader.join(3)
    assert not reader.is_alive()
    assert results == [b"image"]
    assert len(opens) == 1


def test_changed_or_deleted_file_is_never_served_from_cache(cache, tmp_path):
    path = tmp_path / "label.json"
    path.write_bytes(b"old")
    cache.prefetch([path])
    wait_idle(cache)
    original = path.stat()
    path.write_bytes(b"new")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 10**9))
    assert cache.read(path) == b"new"
    # Atomic replacement is detected even with the same size and timestamp.
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"new")
    metadata = path.stat()
    replacement.write_bytes(b"yes")
    os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    os.replace(replacement, path)
    assert cache.read(path) == b"yes"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        cache.read(path)


def test_missing_annotation_can_be_created_after_prefetch(cache, tmp_path):
    path = tmp_path / "new.json"
    cache.prefetch([path])
    wait_idle(cache)
    path.write_bytes(b"created")
    assert cache.read(path) == b"created"


def test_default_budget_holds_a_long_read_ahead_window(cache):
    assert cache.max_bytes == 512 * 1024 * 1024


def test_shrinking_budget_evicts_farthest_files_first(tmp_path):
    files = [tmp_path / f"image-{i}.jpg" for i in range(4)]
    for path in files:
        path.write_bytes(b"1234")
    cache = FilePrefetchCache(max_bytes=16, workers=1)
    try:
        cache.prefetch(files)
        wait_idle(cache)
        assert cache.cached_bytes == 16
        cache.set_max_bytes(8)
        assert cache.max_bytes == 8
        assert cache.cached_bytes == 8
        with mock.patch.object(
            file_prefetch, "open_file", side_effect=AssertionError("reread")
        ):
            assert cache.read(files[0]) == b"1234"
            assert cache.read(files[1]) == b"1234"
        cache.set_max_bytes(16)
        cache.prefetch(files)
        wait_idle(cache)
        assert cache.cached_bytes == 16
    finally:
        cache.close()


def test_is_cached_reports_only_files_held_in_memory(cache, tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"image")
    assert not cache.is_cached(path)
    cache.prefetch([path])
    wait_idle(cache)
    assert cache.is_cached(path)
    assert cache.is_idle()
    cache.clear()
    assert not cache.is_cached(path)


def test_budget_preserves_nearest_file_and_skips_oversized_prefetch(tmp_path):
    near, far, huge = [tmp_path / name for name in ("near", "far", "huge")]
    near.write_bytes(b"next")
    far.write_bytes(b"away")
    huge.write_bytes(b"large image")
    cache = FilePrefetchCache(max_bytes=6, workers=1)
    try:
        with mock.patch.object(
            file_prefetch, "open_file", wraps=file_prefetch.open_file
        ) as opening:
            cache.prefetch([near, far, huge])
            wait_idle(cache)
            assert opening.call_count == 2
        assert cache.cached_bytes == 4
        with mock.patch.object(
            file_prefetch, "open_file", side_effect=AssertionError("reread")
        ):
            assert cache.read(near) == b"next"
        assert cache.read(huge) == b"large image"
        assert cache.cached_bytes <= 6
    finally:
        cache.close()


@pytest.mark.parametrize("close", [False, True])
def test_late_network_results_are_discarded_without_waiting(
    cache, tmp_path, monkeypatch, close
):
    path = tmp_path / "old-folder.jpg"
    path.write_bytes(b"old network data")
    started, release = threading.Event(), threading.Event()
    real_open = file_prefetch.open_file

    def slow_open(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(file_prefetch, "open_file", slow_open)
    cache.prefetch([path])
    try:
        assert started.wait(3)
        if close:
            cache.close()
        else:
            cache.clear()
        assert not release.is_set()
        assert cache.cached_bytes == 0
    finally:
        release.set()
    wait_idle(cache)
    assert cache.cached_bytes == 0


def test_annotations_remain_fresh_and_editable_with_separate_output_directory(
    cache, tmp_path
):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    image = images / "사진.png"
    Image.new("RGB", (3, 2)).save(image)
    label = labels / "사진.json"
    LabelFile().save(
        filename=str(label), shapes=[], image_path=image.name,
        image_height=2, image_width=3, other_data={"checked": False},
    )
    cache.prefetch([image, label])
    wait_idle(cache)
    first = LabelFile(str(label), str(images), read_file=cache.read)
    first.other_data["description"] = "unsaved edit"
    second = LabelFile(str(label), str(images), read_file=cache.read)
    assert second.other_data["description"] == ""
    assert second.image_data == image.read_bytes()
    first.save(
        filename=str(label), shapes=[], image_path=image.name,
        image_height=2, image_width=3,
        other_data={"description": "saved edit", "checked": True},
    )
    updated = LabelFile(str(label), str(images), read_file=cache.read)
    assert updated.other_data["description"] == "saved edit"
    assert updated.other_data["checked"] is True


def test_embedded_image_data_takes_precedence_over_external_image(cache, tmp_path):
    import base64
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (3, 2)).save(buffer, format="PNG")
    label = tmp_path / "image.json"
    label.write_text(json.dumps({
        "version": "test", "shapes": [], "imagePath": "missing.png",
        "imageData": base64.b64encode(buffer.getvalue()).decode(),
    }))
    cache.prefetch([label])
    wait_idle(cache)
    loaded = LabelFile(str(label), read_file=cache.read)
    assert loaded.image_data == buffer.getvalue()
