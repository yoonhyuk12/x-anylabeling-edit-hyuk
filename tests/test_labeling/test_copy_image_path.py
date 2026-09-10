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
                str(workdir / "settings.ini"),
                original_settings.Format.IniFormat,
            )

    monkeypatch.setattr(QtCore, "QSettings", IsolatedSettings)

    folder = tmp_path / "images"
    folder.mkdir()
    for index in range(2):
        Image.new("RGB", (32, 24), (index * 20, 0, 0)).save(
            folder / f"image-{index}.png"
        )

    config = copy.deepcopy(load_template_config())
    config["exif_scan_enabled"] = False
    window = MainWindow(app, config=config)
    window.show()
    app.processEvents()

    yield app, folder, window.labeling_widget.view

    window.labeling_widget.view.dirty = False
    window.close()
    app.processEvents()


def test_action_sits_in_the_file_menu(env):
    _app, _folder, widget = env

    assert widget.actions.copy_image_path in widget.menus.file.actions()


def test_action_is_disabled_until_an_image_is_open(env):
    app, folder, widget = env

    assert not widget.actions.copy_image_path.isEnabled()

    widget.import_image_folder(str(folder))
    app.processEvents()

    assert widget.actions.copy_image_path.isEnabled()


def test_copying_puts_the_current_image_path_on_the_clipboard(env):
    app, folder, widget = env
    widget.import_image_folder(str(folder))
    widget.load_file(str(folder / "image-1.png"))
    app.processEvents()

    copied = []
    widget.copy_file_path = copied.append
    widget.actions.copy_image_path.trigger()

    assert copied == [os.path.normpath(str(folder / "image-1.png"))]


def test_copying_without_an_open_image_does_nothing(env):
    _app, _folder, widget = env

    copied = []
    widget.copy_file_path = copied.append
    widget.copy_current_image_path()

    assert copied == []
