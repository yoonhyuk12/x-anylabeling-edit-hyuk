import copy
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets

from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.utils.folder_scan import scan_label_classes
from anylabeling.views.labeling.widgets.label_dialog import LabelModifyDialog
from anylabeling.views.labeling.widgets.unique_label_qlist_widget import (
    UniqueLabelQListWidget,
)


class TestLabelManagerLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (
            QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.images = [str(self.folder / f"사진{i}.jpg") for i in range(3)]
        for index, labels in enumerate((["cat", "dog"], ["dog"])):
            (self.folder / f"사진{index}.json").write_text(
                json.dumps({"shapes": [{"label": s} for s in labels]}),
                encoding="utf-8",
            )
        self.parent = QtWidgets.QWidget()
        self.addCleanup(self.parent.close)
        self.parent.file_list_widget = QtWidgets.QListWidget(self.parent)
        self.parent.file_list_widget.addItems(self.images)
        self.parent.unique_label_list = UniqueLabelQListWidget(self.parent)
        self.parent.output_dir = None
        self.parent.label_info = {
            "cat": dict(
                delete=False, value=None, color=[1, 2, 3],
                opacity=90, visible=False,
            )
        }
        self.parent._get_rgb_by_label = lambda _: (4, 5, 6)
        self.parent.filename = self.images[0]
        self.parent.load_file = mock.Mock()

    def test_scan_reports_completed_files_including_missing_annotations(self):
        progress = []
        result = scan_label_classes(
            self.images, cancel=threading.Event(),
            report=lambda n, total: progress.append((n, total)),
        )
        self.assertEqual(result, {"cat", "dog"})
        self.assertEqual(progress, [(0, 3), (1, 3), (2, 3), (3, 3)])

    def test_scan_uses_selected_annotation_directory(self):
        output = self.folder / "labels"
        output.mkdir()
        (output / "사진0.json").write_text(
            '{"shapes": [{"label": "horse"}]}', encoding="utf-8"
        )
        result = scan_label_classes(
            self.images, str(output), cancel=threading.Event(),
            report=lambda *_: None,
        )
        self.assertEqual(result, {"horse"})

    def test_loaded_manager_preserves_colors_and_visibility(self):
        original = {
            p: p.read_bytes() for p in self.folder.glob("*.json")
        }
        dialog = LabelModifyDialog(self.parent)
        self.addCleanup(dialog.close)
        self.assertTrue(dialog.labels_loaded)
        self.assertEqual(dialog.table_widget.rowCount(), 2)
        self.assertEqual(self.parent.label_info["cat"]["color"], [1, 2, 3])
        self.assertEqual(self.parent.label_info["cat"]["opacity"], 90)
        self.assertFalse(self.parent.label_info["cat"]["visible"])
        self.assertTrue(self.parent.label_info["dog"]["visible"])
        self.assertEqual(len(self.parent.unique_label_list.find_items_by_label(
            "dog"
        )), 1)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)

    def test_loading_visible_and_cancel_works_during_blocked_network_read(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        original_load = json.load
        original_info = copy.deepcopy(self.parent.label_info)
        observed_progress = []

        def slow_load(stream):
            started.set()
            try:
                release.wait(5)
                return original_load(stream)
            finally:
                finished.set()

        def cancel_when_visible():
            dialog = self.app.activeModalWidget()
            if isinstance(dialog, QtWidgets.QProgressDialog):
                if started.is_set() and "0 / 3" in dialog.labelText():
                    bar = dialog.findChild(QtWidgets.QProgressBar)
                    observed_progress.append((
                        dialog.labelText(), bar.value(), bar.maximum()
                    ))
                    dialog.findChild(QtWidgets.QPushButton).click()

        timer = QtCore.QTimer()
        timer.timeout.connect(cancel_when_visible)
        timer.start(20)
        try:
            with mock.patch(
                "anylabeling.views.labeling.utils.folder_scan.json.load",
                side_effect=slow_load,
            ), mock.patch.object(LabelModifyDialog, "exec") as show_manager:
                LabelingWidget.label_manager(self.parent)
                show_manager.assert_not_called()
                self.assertTrue(observed_progress)
                self.assertEqual(observed_progress[0][1:], (0, 3))
                self.assertFalse(finished.is_set())
                self.assertEqual(self.parent.label_info, original_info)
                self.assertEqual(self.parent.unique_label_list.count(), 0)
                self.parent.load_file.assert_not_called()
        finally:
            timer.stop()
            release.set()
        self.assertTrue(finished.wait(2))
        self.app.processEvents()
        self.assertEqual(self.parent.label_info, original_info)

    def test_bad_json_shows_error_without_opening_partial_manager(self):
        (self.folder / "사진1.json").write_text("{broken", encoding="utf-8")
        original_info = copy.deepcopy(self.parent.label_info)
        with mock.patch.object(
            QtWidgets.QMessageBox, "critical"
        ) as error, mock.patch.object(LabelModifyDialog, "exec") as show:
            LabelingWidget.label_manager(self.parent)
        error.assert_called_once()
        show.assert_not_called()
        self.assertEqual(self.parent.label_info, original_info)
        self.assertEqual(self.parent.unique_label_list.count(), 0)

    def test_empty_folder_still_opens_manager(self):
        self.parent.file_list_widget.clear()
        dialog = LabelModifyDialog(self.parent)
        self.addCleanup(dialog.close)
        self.assertTrue(dialog.labels_loaded)
        self.assertEqual(dialog.table_widget.rowCount(), 1)


if __name__ == "__main__":
    unittest.main()
