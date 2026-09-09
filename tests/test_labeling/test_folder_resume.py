import copy
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PyQt6 import QtCore, QtWidgets

from anylabeling import config as app_config
from anylabeling.resources import resources  # noqa: F401
from anylabeling.views.mainwindow import MainWindow
from anylabeling.views.labeling.settings.schema import load_template_config


@pytest.fixture
def env(tmp_path, monkeypatch):
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

    folder = tmp_path / "images"
    folder.mkdir()
    for index in range(5):
        Image.new("RGB", (32, 24), (index * 20, 0, 0)).save(
            folder / f"image-{index}.png"
        )

    def make_window():
        config = copy.deepcopy(load_template_config())
        config["exif_scan_enabled"] = False
        window = MainWindow(app, config=config)
        window.show()
        app.processEvents()
        return window

    def close_window(window):
        window.labeling_widget.view.dirty = False
        window.close()
        app.processEvents()

    return app, folder, make_window, close_window


def test_reopening_folder_resumes_last_viewed_image(env):
    app, folder, make_window, close_window = env

    window = make_window()
    widget = window.labeling_widget.view
    widget.import_image_folder(str(folder))
    assert widget.filename == str(folder / "image-0.png")
    widget.load_file(str(folder / "image-3.png"))
    app.processEvents()
    assert widget.filename == str(folder / "image-3.png")
    close_window(window)

    window = make_window()
    widget = window.labeling_widget.view
    widget.import_image_folder(str(folder))
    app.processEvents()
    assert widget.filename == str(folder / "image-3.png")
    assert widget.file_list_widget.currentRow() == 3
    close_window(window)


def test_missing_remembered_image_falls_back_to_first(env):
    app, folder, make_window, close_window = env

    window = make_window()
    widget = window.labeling_widget.view
    widget.import_image_folder(str(folder))
    widget.load_file(str(folder / "image-4.png"))
    app.processEvents()
    close_window(window)

    (folder / "image-4.png").unlink()

    window = make_window()
    widget = window.labeling_widget.view
    widget.import_image_folder(str(folder))
    app.processEvents()
    assert widget.filename == str(folder / "image-0.png")
    close_window(window)


def test_each_folder_keeps_its_own_position(env, tmp_path):
    app, folder, make_window, close_window = env
    other = tmp_path / "other"
    other.mkdir()
    for index in range(3):
        Image.new("RGB", (16, 16), (0, index * 40, 0)).save(
            other / f"pic-{index}.png"
        )

    window = make_window()
    widget = window.labeling_widget.view
    widget.import_image_folder(str(folder))
    widget.load_file(str(folder / "image-2.png"))
    app.processEvents()
    widget.import_image_folder(str(other))
    widget.load_file(str(other / "pic-1.png"))
    app.processEvents()
    close_window(window)

    window = make_window()
    widget = window.labeling_widget.view
    widget.import_image_folder(str(folder))
    app.processEvents()
    assert widget.filename == str(folder / "image-2.png")
    widget.import_image_folder(str(other))
    app.processEvents()
    assert widget.filename == str(other / "pic-1.png")
    close_window(window)
