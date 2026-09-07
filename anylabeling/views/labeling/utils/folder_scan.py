"""Read folder metadata without downloading every annotation or image."""

import json
import os
import os.path as osp
import queue
import threading
from dataclasses import dataclass, field

import natsort
from PyQt6 import QtCore, QtWidgets

from ._io import open_file
from .file_search import (
    matches_filename,
    matches_label_attribute,
    parse_search_pattern,
)


@dataclass
class FolderContents:
    image_files: list
    label_files: set
    review_statuses: dict = field(default_factory=dict)


def label_path(image_file, output_dir=None):
    path = osp.splitext(image_file)[0] + ".json"
    if output_dir:
        path = osp.join(output_dir, osp.basename(path))
    return osp.normpath(osp.abspath(path))


def scan_label_classes(image_files, output_dir=None, *, cancel, report):
    """Collect labels without touching Qt or modifying the current project."""
    classes = set()
    total = len(image_files)
    report(0, total)
    for index, image_file in enumerate(image_files, 1):
        if cancel.is_set():
            return None
        path = label_path(image_file, output_dir)
        try:
            with open_file(path, "r", encoding="utf-8") as stream:
                data = json.load(stream)
        except FileNotFoundError:
            # Images without annotations still count towards scan progress.
            report(index, total)
            continue
        for shape in data.get("shapes", []):
            if cancel.is_set():
                return None
            classes.add(shape["label"])
        report(index, total)
    return None if cancel.is_set() else classes


def scan_image_folder(
    folder,
    extensions,
    output_dir=None,
    pattern=None,
    *,
    cancel,
    report,
    read_checked=None,
):
    """Read names, optionally reading review status or searching attributes."""
    images = []
    labels = set()
    folder = osp.normpath(osp.abspath(folder))
    extensions = tuple(extensions)

    def on_error(error):
        raise error

    for root, _, files in os.walk(folder, onerror=on_error):
        if cancel.is_set():
            return None
        for name in files:
            if cancel.is_set():
                return None
            path = osp.normpath(osp.join(root, name))
            if name.lower().endswith(extensions):
                images.append(path)
                if len(images) % 250 == 0:
                    report(len(images), 0)
            elif not output_dir and name.lower().endswith(".json"):
                labels.add(osp.normcase(path))

    if output_dir:
        try:
            with os.scandir(output_dir) as entries:
                for entry in entries:
                    if cancel.is_set():
                        return None
                    if entry.name.lower().endswith(".json"):
                        labels.add(osp.normcase(osp.abspath(entry.path)))
        except FileNotFoundError:
            pass

    if cancel.is_set():
        return None
    try:
        images = natsort.natsorted(images)
    except (OSError, ValueError):
        images.sort()

    if pattern:
        search = parse_search_pattern(pattern)
        filtered = []
        for index, filename in enumerate(images, 1):
            if cancel.is_set():
                return None
            if index % 100 == 0 or search.mode == "attribute":
                report(index, len(images))
            if search.mode == "index":
                if index == search.index:
                    filtered.append(filename)
                    break
            elif matches_filename(filename, search):
                if search.mode != "attribute" or matches_label_attribute(
                    filename, label_path(filename, output_dir), search
                ):
                    filtered.append(filename)
        images = filtered
    review_statuses = {}
    if read_checked is not None:
        for index, filename in enumerate(images, 1):
            if cancel.is_set():
                return None
            report(index, len(images))
            path = label_path(filename, output_dir)
            review_statuses[filename] = (
                read_checked(path) if osp.normcase(path) in labels else False
            )
    return FolderContents(images, labels, review_statuses)


def run_cancellable_task(parent, title, operation):
    """Keep Qt responsive while a read-only filesystem operation runs.

    A cancelled network read may finish later. The daemon thread owns no Qt
    objects and its result is discarded, so closing the dialog never waits for
    the network and cannot update a subsequently opened folder.
    """
    cancel = threading.Event()
    updates = queue.SimpleQueue()
    results = queue.SimpleQueue()

    def work():
        try:
            value = operation(cancel, lambda n, total: updates.put((n, total)))
            results.put((value, None))
        except Exception as error:
            results.put((None, error))

    dialog = QtWidgets.QProgressDialog(
        title, parent.tr("Cancel"), 0, 0, parent
    )
    dialog.setWindowTitle(title)
    dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    progress_bar = QtWidgets.QProgressBar(dialog)
    progress_bar.setRange(0, 0)
    dialog.setBar(progress_bar)
    dialog.canceled.connect(cancel.set)
    outcome = []

    def poll():
        latest = None
        while not updates.empty():
            latest = updates.get()
        if latest is not None:
            count, total = latest
            suffix = f"{count:,} / {total:,}" if total else f"{count:,}"
            dialog.setLabelText(f"{title}\n{suffix}")
            # Update the bar directly: modal QProgressDialog.setValue()
            # processes events recursively while this timer is polling.
            progress_bar.setRange(0, total)
            if total:
                progress_bar.setValue(count)
        if not results.empty():
            outcome.append(results.get())
            dialog.accept()

    timer = QtCore.QTimer(dialog)
    timer.timeout.connect(poll)
    timer.start(40)
    threading.Thread(target=work, daemon=True, name="folder-reader").start()
    try:
        dialog.exec()
    finally:
        cancel.set()
        timer.stop()
        dialog.deleteLater()
    if not outcome:
        return None
    value, error = outcome[0]
    if error is not None:
        raise error
    return value
