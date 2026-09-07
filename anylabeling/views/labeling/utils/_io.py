import contextlib
import os


def io_path(path):
    """Use Windows extended paths for long filesystem paths only.

    Keep these paths at the I/O boundary so display names and imagePath values
    in annotation files remain portable.
    """
    path = os.fspath(path)
    if os.name != "nt" or not isinstance(path, str):
        return path
    if path.startswith(("\\\\?\\", "\\\\.\\")):
        return path
    absolute = os.path.abspath(path)
    if len(absolute) < 260:
        return path
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def open_file(path, *args, **kwargs):
    return open(io_path(path), *args, **kwargs)


@contextlib.contextmanager
def io_open(name, mode):
    assert mode in ["r", "w"]
    encoding = "utf-8"
    with open_file(name, mode, encoding=encoding) as stream:
        yield stream
