"""verl.utils.hdfs_io logs a failed local copy with its traceback (audit LOG006)."""
import logging

from verl.utils import hdfs_io


def test_failed_local_copy_logs_traceback(tmp_path, caplog):
    missing = tmp_path / "missing.txt"

    with caplog.at_level(logging.WARNING):
        ok = hdfs_io._copy(str(missing), str(tmp_path / "dst.txt"))

    assert ok is False
    [record] = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert record.exc_info is not None
    assert record.exc_info[0] is FileNotFoundError
