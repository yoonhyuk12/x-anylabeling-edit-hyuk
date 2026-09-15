import copy
import os
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PyQt6 import QtCore, QtWidgets

from anylabeling import config as app_config
from anylabeling.resources import resources  # noqa: F401
from anylabeling.views.mainwindow import MainWindow
from anylabeling.views.labeling.label_file import LabelFile
from anylabeling.views.labeling.settings.schema import load_template_config
from anylabeling.views.labeling.utils import file_prefetch


@pytest.fixture
def navigation(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    workdir = tmp_path / "config"
    workdir.mkdir()
    monkeypatch.setattr(app_config, "_work_directory", str(workdir))
    monkeypatch.setattr(app_config, "current_config_file", "{}")
    original_settings = QtCore.QSettings

    class IsolatedSettings(original_settings):
        def __init__(self, *_args, **_kwargs):
            super().__init__(
                str(workdir / "settings.ini"), original_settings.Format.IniFormat
            )

    monkeypatch.setattr(QtCore, "QSettings", IsolatedSettings)
    config = copy.deepcopy(load_template_config())
    config["exif_scan_enabled"] = False
    window = MainWindow(app, config=config)
    window.show()
    app.processEvents()
    widget = window.labeling_widget.view
    folder = tmp_path / "images"
    folder.mkdir()
    for index in range(7):
        image = folder / f"image-{index}.png"
        Image.new("RGB", (32, 24), (index * 20, 0, 0)).save(image)
        LabelFile().save(
            filename=str(image.with_suffix(".json")), shapes=[],
            image_path=image.name, image_width=32, image_height=24,
            other_data={"checked": False},
        )
    try:
        widget.import_image_folder(str(folder))
        yield app, widget, folder
    finally:
        widget.dirty = False
        window.close()
        app.processEvents()


def wait_prefetch(widget):
    cache = widget._file_prefetcher
    with cache._condition:
        assert cache._condition.wait_for(
            lambda: not cache._pending and not cache._inflight, timeout=5
        )


def test_next_and_previous_actions_use_prefetched_images_and_labels(navigation):
    app, widget, folder = navigation
    widget._config["image_prefetch_count"] = 5
    widget._prefetch_neighbor_files()
    wait_prefetch(widget)
    assert widget.filename == str(folder / "image-0.png")
    original_read = file_prefetch.open_file
    reads = []

    def record_open(path, *args, **kwargs):
        reads.append(os.path.basename(path))
        return original_read(path, *args, **kwargs)

    with mock.patch.object(file_prefetch, "open_file", side_effect=record_open):
        widget.actions.open_next_image.trigger()
        app.processEvents()
        assert widget.filename == str(folder / "image-1.png")
        assert widget.image.pixelColor(0, 0).red() == 20
        wait_prefetch(widget)
        # Only the new farthest neighbor needs reading after pressing D.
        assert sorted(reads) == ["image-6.json", "image-6.png"]
        reads.clear()
        widget.actions.open_prev_image.trigger()
        app.processEvents()
        assert widget.filename == str(folder / "image-0.png")
        assert widget.image.pixelColor(0, 0).red() == 0
        wait_prefetch(widget)
        assert reads == []


def test_next_image_sees_external_annotation_changes(navigation):
    app, widget, folder = navigation
    wait_prefetch(widget)
    label = folder / "image-1.json"
    LabelFile().save(
        filename=str(label), shapes=[], image_path="image-1.png",
        image_height=24, image_width=32,
        other_data={"checked": True, "description": "new external annotation"},
    )
    widget.actions.open_next_image.trigger()
    app.processEvents()
    assert widget.other_data["checked"] is True
    assert widget.other_data["description"] == "new external annotation"


def test_read_ahead_setting_applies_immediately_and_can_be_disabled(navigation):
    _, widget, folder = navigation
    wait_prefetch(widget)
    widget._config["image_prefetch_count"] = 0
    widget._settings_runtime_applier.apply_change("image_prefetch_count", 0)
    assert widget._file_prefetcher.cached_bytes == 0
    widget._config["image_prefetch_count"] = 2
    with mock.patch.object(widget._file_prefetcher, "prefetch") as prefetch:
        widget._settings_runtime_applier.apply_change("image_prefetch_count", 2)
    assert prefetch.call_args.args[0] == [
        str(folder / name)
        for name in (
            "image-1.png", "image-1.json", "image-2.png", "image-2.json",
            "image-0.png", "image-0.json",
        )
    ]


def test_read_ahead_cache_size_setting_applies_immediately(navigation):
    _, widget, _ = navigation
    wait_prefetch(widget)
    assert widget._file_prefetcher.max_bytes == 512 * 1024 * 1024
    widget._config["image_prefetch_cache_mb"] = 64
    widget._settings_runtime_applier.apply_change(
        "image_prefetch_cache_mb", 64
    )
    assert widget._file_prefetcher.max_bytes == 64 * 1024 * 1024
    assert widget._file_prefetcher.cached_bytes > 0


def test_read_ahead_window_defaults_to_twenty_and_is_capped_at_hundred(
    navigation,
):
    _, widget, folder = navigation
    wait_prefetch(widget)
    assert widget._config["image_prefetch_count"] == 20
    widget._config["image_prefetch_count"] = 1000
    with mock.patch.object(widget._file_prefetcher, "prefetch") as prefetch:
        widget._settings_runtime_applier.apply_change(
            "image_prefetch_count", 1000
        )
    paths = prefetch.call_args.args[0]
    # Seven images: six neighbours plus the current one, image and json each.
    assert len(paths) == 14
    assert paths[0] == str(folder / "image-1.png")


def test_read_ahead_indicator_shows_progress_in_top_right_corner(navigation):
    app, widget, _ = navigation
    wait_prefetch(widget)
    widget._update_prefetch_indicator()
    app.processEvents()
    indicator = widget.prefetch_indicator
    assert indicator.isVisible()
    # Six images follow image-0, and all of them are already in memory.
    assert indicator.text() == "Read ahead 6 / 6"
    assert indicator.property("complete") is True
    viewport = widget._canvas_scroll_area.viewport()
    assert indicator.x() + indicator.width() == viewport.width() - 10
    assert indicator.y() == 10
    widget._config["image_prefetch_count"] = 0
    widget._settings_runtime_applier.apply_change("image_prefetch_count", 0)
    app.processEvents()
    assert not indicator.isVisible()


def test_read_ahead_indicator_counts_only_finished_reads(navigation):
    app, widget, folder = navigation
    wait_prefetch(widget)
    cache = widget._file_prefetcher
    cache.clear()
    only_first = lambda path: path.endswith("image-1.png")  # noqa: E731
    with mock.patch.object(cache, "is_cached", side_effect=only_first):
        widget._update_prefetch_indicator()
    assert widget.prefetch_indicator.text() == "Read ahead 1 / 6"
    assert widget.prefetch_indicator.property("complete") is False
