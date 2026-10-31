"""Where DeepRetrieval's datasets and retrieval indexes live.

Everything that used to be a cwd-relative `data/...` path goes through `data_path`. The root is
`$DEEPRETRIEVAL_DATA_DIR` when set (e.g. a network volume mounted on a GPU pod), else
`<code>/data`, resolved from this file's location so it no longer depends on the cwd.
"""
import os
from pathlib import Path

DATA_DIR_ENV = "DEEPRETRIEVAL_DATA_DIR"

# verl/utils/paths.py -> verl/utils -> verl -> code/
CODE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = CODE_DIR / "data"

# Paths stored inside datasets (e.g. SQL `extra_info.db_path`) were written relative to code/.
_STORED_DATA_PREFIXES = ("./data/", "data/")


def data_dir() -> Path:
    """The data root: $DEEPRETRIEVAL_DATA_DIR (``~`` expanded, made absolute) or ``<code>/data``."""
    configured = os.environ.get(DATA_DIR_ENV, "").strip()
    if not configured:
        return DEFAULT_DATA_DIR
    return Path(configured).expanduser().absolute()


def data_path(*parts: str) -> Path:
    """``data_dir() / parts``; each part is relative and may contain ``/`` separators."""
    root = data_dir()
    for part in parts:
        if Path(part).is_absolute():
            raise ValueError(f"data_path parts must be relative, got {part!r}")
    result = root.joinpath(*parts)
    if os.path.normpath(result) != str(root) and not os.path.normpath(result).startswith(str(root) + os.sep):
        raise ValueError(f"data_path{parts!r} resolves outside the data dir {root}")
    return result


def resolve_data_reference(stored: str) -> Path:
    """Map a path stored in a dataset to a real path.

    ``data/...`` (how the preprocessing scripts write paths) is re-rooted under `data_dir()`;
    absolute paths and any other relative path are returned unchanged.
    """
    for prefix in _STORED_DATA_PREFIXES:
        if stored.startswith(prefix):
            return data_path(stored[len(prefix):])
    return Path(stored)


def stored_data_reference(relpath: str) -> str:
    """The machine-independent form of a data file for storing inside a dataset (``data/<relpath>``).

    Preprocessing writes this (not an absolute path) so parquet files stay valid when the data dir
    moves; readers turn it back into a real path with `resolve_data_reference`.
    """
    if Path(relpath).is_absolute():
        raise ValueError(f"stored data references must be relative, got {relpath!r}")
    return _STORED_DATA_PREFIXES[1] + relpath


def require_existing(path: Path, what: str) -> Path:
    """Return `path`, or raise FileNotFoundError naming it and the data-dir setting."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path} (data dir {data_dir()}; set ${DATA_DIR_ENV} to change it, "
            f"default {DEFAULT_DATA_DIR})")
    return path
