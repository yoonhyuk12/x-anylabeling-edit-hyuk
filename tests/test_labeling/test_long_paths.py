import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from anylabeling.views.labeling.label_file import LabelFile, LabelFileError
from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.utils._io import io_open, io_path, open_file
from anylabeling.views.labeling.utils.file_search import (
    matches_label_attribute,
    parse_search_pattern,
)
from anylabeling.views.labeling.utils.folder_scan import scan_label_classes


@unittest.skipUnless(os.name == "nt", "Windows extended paths are required")
class TestWindowsLongPaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "long-path-labels"
        self.directory.mkdir()
        stem = "sample_" + "a" * 220
        self.image_path = self.directory / (stem + ".png")
        self.label_path = self.directory / (stem + ".json")
        for path in (self.image_path, self.label_path):
            self.assertGreater(len(str(path)), 260)
            self.addCleanup(self.remove_file, path)
        with open_file(self.image_path, "wb") as stream:
            Image.new("RGB", (3, 2), "white").save(stream, format="PNG")
        self.data = {
            "version": "test", "flags": {}, "checked": True,
            "shapes": [{"label": "person", "shape_type": "point",
                        "points": [[1, 1]]}],
            "imagePath": self.image_path.name, "imageData": None,
            "imageHeight": 2, "imageWidth": 3,
        }
        with open_file(self.label_path, "w", encoding="utf-8") as stream:
            json.dump(self.data, stream)

    @staticmethod
    def remove_file(path):
        try:
            os.remove(io_path(path))
        except FileNotFoundError:
            pass

    def test_long_image_and_label_open_scan_search_and_save(self):
        label = LabelFile(str(self.label_path))
        self.assertEqual(len(label.shapes), 1)
        self.assertEqual(label.shapes[0].label, "person")
        self.assertTrue(label.image_data)
        self.assertTrue(LabelingWidget._label_file_checked(str(self.label_path)))
        self.assertEqual(scan_label_classes(
            [str(self.image_path)], cancel=threading.Event(),
            report=lambda *_: None,
        ), {"person"})
        self.assertTrue(matches_label_attribute(
            str(self.image_path), str(self.label_path),
            parse_search_pattern("label::person"),
        ))

        label.save(
            filename=str(self.label_path), shapes=self.data["shapes"],
            image_path=self.image_path.name, image_height=2, image_width=3,
            other_data={"checked": False},
        )
        loaded = LabelFile(str(self.label_path))
        self.assertFalse(loaded.other_data["checked"])
        self.assertEqual(loaded.filename, str(self.label_path))
        self.assertEqual(loaded.image_path, self.image_path.name)
        with open_file(self.label_path, encoding="utf-8") as stream:
            saved = json.load(stream)
        self.assertEqual(saved["imagePath"], self.image_path.name)
        self.assertEqual(saved["shapes"], self.data["shapes"])

    def test_failed_atomic_save_preserves_long_path_file(self):
        with open_file(self.label_path, "rb") as stream:
            original = stream.read()
        with mock.patch(
            "anylabeling.views.labeling.label_file.os.replace",
            side_effect=OSError("simulated replace failure"),
        ):
            with self.assertRaises(LabelFileError):
                LabelFile().save(
                    filename=str(self.label_path), shapes=[],
                    image_path=self.image_path.name,
                )
        with open_file(self.label_path, "rb") as stream:
            self.assertEqual(stream.read(), original)
        self.assertEqual(set(os.listdir(self.directory)),
                         {self.label_path.name, self.image_path.name})

    def test_io_open_closes_long_path_handles_even_on_error(self):
        with self.assertRaises(ValueError):
            with io_open(self.label_path, "r") as stream:
                self.assertEqual(json.load(stream)["checked"], True)
                raise ValueError("stop reading")
        self.assertTrue(stream.closed)

    def test_unc_and_already_extended_paths(self):
        unc = "\\\\server\\share\\" + "segment\\" * 40 + "label.json"
        self.assertEqual(io_path(unc), "\\\\?\\UNC\\" + unc[2:])
        extended = io_path(self.label_path)
        self.assertTrue(extended.startswith("\\\\?\\"))
        self.assertEqual(io_path(extended), extended)
        self.assertEqual(io_path("relative.json"), "relative.json")


if __name__ == "__main__":
    unittest.main()
