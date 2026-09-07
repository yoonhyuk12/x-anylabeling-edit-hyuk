import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.utils.folder_scan import (
    label_path,
    run_cancellable_task,
    scan_image_folder,
)


class FolderHarness(QtWidgets.QWidget):
    import_image_folder = LabelingWidget.import_image_folder
    _create_file_list_item = LabelingWidget._create_file_list_item
    _set_file_item_checked = LabelingWidget._set_file_item_checked
    _label_file_checked = staticmethod(LabelingWidget._label_file_checked)
    _open_unchecked_image = LabelingWidget._open_unchecked_image
    image_list = LabelingWidget.image_list

    def __init__(self):
        super().__init__()
        self._config = {"exif_scan_enabled": False}
        self.output_dir = None
        self.filename = "old.jpg"
        self.last_open_dir = "old-folder"
        self.file_list_widget = QtWidgets.QListWidget(self)
        self.file_list_widget.setUniformItemSizes(True)
        self.file_list_widget.addItem(self.filename)
        self.fn_to_index = {self.filename: 0}
        self.file_status_icons = {True: QtGui.QIcon(), False: QtGui.QIcon()}
        self.compare_view_manager = mock.Mock()
        self.compare_view_manager.is_active.return_value = False
        self.actions = SimpleNamespace(**{
            name: QtGui.QAction(self) for name in (
                "open_next_image", "open_prev_image",
                "open_next_unchecked_image", "open_prev_unchecked_image",
            )
        })
        self.may_continue = lambda: True
        self.open_next_image = mock.Mock()
        self.load_file = mock.Mock()
        self.toggle_actions = mock.Mock()
        self.async_exif_scanner = mock.Mock()
        self.error_message = mock.Mock()
        self._file_prefetcher = mock.Mock()


class TestFolderLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (
            QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        for name in ("사진10.JPG", "사진2.jpg", "사진1.jpg"):
            (self.folder / name).touch()
        (self.folder / "사진1.json").write_text('{"checked": true}')
        (self.folder / "사진2.json").write_text('{"checked": false}')
        self.widget = FolderHarness()
        self.addCleanup(self.widget.close)

    def scan(self, **kwargs):
        return scan_image_folder(
            self.folder, (".jpg",), cancel=threading.Event(),
            report=lambda *_: None, **kwargs,
        )

    def test_folder_open_reads_names_without_opening_images_or_json(self):
        with mock.patch(
            "builtins.open", side_effect=AssertionError("content read")
        ):
            self.widget.import_image_folder(str(self.folder), load=False)
        self.widget.error_message.assert_not_called()
        self.assertEqual(
            [Path(f).name for f in self.widget.image_list],
            ["사진1.jpg", "사진2.jpg", "사진10.JPG"],
        )
        self.assertNotIn("old.jpg", self.widget.fn_to_index)
        self.widget.open_next_image.assert_called_once_with(load=False)
        self.widget.async_exif_scanner.start_scan.assert_not_called()
        first = self.widget.file_list_widget.item(0)
        self.assertIsNone(first.data(QtCore.Qt.ItemDataRole.UserRole))
        self.assertEqual(first.checkState(), QtCore.Qt.CheckState.Checked)
        last = self.widget.file_list_widget.item(2)
        self.assertIs(last.data(QtCore.Qt.ItemDataRole.UserRole), False)
        self.assertEqual(last.checkState(), QtCore.Qt.CheckState.Unchecked)

    def test_recursive_scan_search_and_separate_output_directory(self):
        sub = self.folder / "sub"
        sub.mkdir()
        (sub / "image.jpg").touch()
        self.assertEqual(len(self.scan().image_files), 4)
        self.assertEqual(
            Path(self.scan(pattern="#3").image_files[0]).name, "사진2.jpg"
        )
        self.assertEqual(
            len(self.scan(pattern="<사진[12]\\.jpg$>").image_files), 2
        )
        output = self.folder / "labels"
        output.mkdir()
        (output / "사진2.json").write_text('{"checked": true}')
        result = self.scan(output_dir=str(output), pattern="checked::1")
        self.assertEqual(
            [Path(f).name for f in result.image_files], ["사진2.jpg"]
        )
        self.assertEqual(result.label_files, {
            os.path.normcase(
                label_path(str(self.folder / "사진2.jpg"), str(output))
            )
        })

    def test_disabling_fast_loading_restores_all_review_statuses(self):
        self.widget._config["fast_folder_loading"] = False
        self.widget.import_image_folder(str(self.folder), load=False)
        self.widget.error_message.assert_not_called()
        self.assertEqual(
            [self.widget.file_list_widget.item(i).data(
                QtCore.Qt.ItemDataRole.UserRole
            ) for i in range(3)],
            [True, False, False],
        )
        self.widget._config["fast_folder_loading"] = True
        with mock.patch(
            "builtins.open", side_effect=AssertionError("content read")
        ):
            self.widget.import_image_folder(str(self.folder), load=False)
        self.widget.error_message.assert_not_called()
        self.assertIsNone(self.widget.file_list_widget.item(0).data(
            QtCore.Qt.ItemDataRole.UserRole
        ))

    def test_full_review_scan_uses_output_labels_and_can_be_cancelled(self):
        output = self.folder / "labels"
        output.mkdir()
        (output / "사진2.json").write_text('{"checked": true}')
        contents = self.scan(
            output_dir=str(output),
            read_checked=LabelingWidget._label_file_checked,
        )
        self.assertEqual(list(contents.review_statuses.values()),
                         [False, True, False])

        cancel = threading.Event()
        reads = []

        def read_checked(path):
            reads.append(path)
            cancel.set()
            return True

        result = scan_image_folder(
            self.folder, (".jpg",), cancel=cancel,
            report=lambda *_: None, read_checked=read_checked,
        )
        self.assertIsNone(result)
        self.assertEqual(len(reads), 1)

    def test_unchecked_navigation_resolves_status_in_both_directions(self):
        self.widget.import_image_folder(str(self.folder), load=False)
        files = self.widget.image_list
        self.widget.filename = files[2]
        self.widget._open_unchecked_image(-1)
        self.widget.load_file.assert_called_with(files[1])
        # Skip the checked first image even before it has been opened.
        self.widget.load_file.reset_mock()
        self.widget.filename = files[1]
        self.widget._open_unchecked_image(-1)
        self.widget.load_file.assert_not_called()
        self.assertIs(
            self.widget.file_list_widget.item(0).data(
                QtCore.Qt.ItemDataRole.UserRole
            ),
            True,
        )
        self.widget.filename = files[0]
        self.widget._open_unchecked_image(1)
        self.widget.load_file.assert_called_with(files[1])

    def test_cancelled_or_failed_import_preserves_existing_folder(self):
        with mock.patch(
            "anylabeling.views.labeling.label_widget.run_cancellable_task",
            return_value=None,
        ):
            self.widget.import_image_folder(str(self.folder))
        self.widget.import_image_folder(str(self.folder / "missing"))
        self.widget.error_message.assert_called_once()
        self.assertEqual(self.widget.image_list, ["old.jpg"])
        self.assertEqual(self.widget.fn_to_index, {"old.jpg": 0})
        self.assertEqual(self.widget.filename, "old.jpg")
        self.assertEqual(self.widget.last_open_dir, "old-folder")

    def test_cancel_returns_while_network_read_is_still_blocked(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        ticks = []

        def operation(cancel, report):
            started.set()
            release.wait(5)
            finished.set()
            return "late result"

        def cancel_dialog():
            ticks.append(True)
            if started.is_set():
                dialog = self.app.activeModalWidget()
                if isinstance(dialog, QtWidgets.QProgressDialog):
                    dialog.findChild(QtWidgets.QPushButton).click()

        timer = QtCore.QTimer()
        timer.timeout.connect(cancel_dialog)
        timer.start(20)
        try:
            result = run_cancellable_task(
                self.widget, "Reading folder", operation
            )
            self.assertIsNone(result)
            self.assertTrue(ticks)
            self.assertFalse(finished.is_set())
        finally:
            timer.stop()
            release.set()
        self.assertTrue(finished.wait(2))
        self.app.processEvents()
        self.assertEqual(self.widget.image_list, ["old.jpg"])

    def test_scan_respects_cancellation_and_exif_option_is_unchanged(self):
        cancel = threading.Event()
        cancel.set()
        self.assertIsNone(scan_image_folder(
            self.folder, (".jpg",), cancel=cancel, report=lambda *_: None
        ))
        self.widget._config["exif_scan_enabled"] = True
        self.widget.import_image_folder(str(self.folder), load=False)
        self.widget.async_exif_scanner.start_scan.assert_called_once_with(
            self.widget.image_list
        )


if __name__ == "__main__":
    unittest.main()
