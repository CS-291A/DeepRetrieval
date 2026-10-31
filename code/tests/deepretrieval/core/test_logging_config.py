"""Logging configuration (audit LOG005): importing verl must not configure logging; entry points call configure_logging."""
import ast
import contextlib
import logging
import subprocess
import sys

import pytest

from tests.deepretrieval.core.conftest import CODE_DIR

_FORMAT_PREFIX = "INFO:"


@pytest.fixture
def restore_levels():
    """Put back the levels configure_logging changes, so other tests see the default tree."""
    loggers = [logging.getLogger(), logging.getLogger("verl"), logging.getLogger("__main__")]
    saved = [lg.level for lg in loggers]
    yield
    for lg, level in zip(loggers, saved, strict=True):
        lg.setLevel(level)


@contextlib.contextmanager
def bare_root():
    """Run with no handlers on the root logger, as in a process nobody has configured.

    pytest attaches its capture handlers to root for each test phase, so this has to run inside
    the test body; on exit every handler added inside is removed and pytest's are put back.
    """
    root = logging.getLogger()
    saved = root.handlers[:]
    for handler in saved:
        root.removeHandler(handler)
    try:
        yield root
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved:
            root.addHandler(handler)


def test_importing_verl_leaves_root_logger_unconfigured():
    probe = (
        "import logging, verl; r = logging.getLogger(); "
        "print(len(r.handlers), logging.getLevelName(r.level))"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=CODE_DIR, capture_output=True, text=True, check=True, timeout=120
    )

    assert result.stdout.split() == ["0", "WARNING"]


@pytest.mark.usefixtures("restore_levels")
def test_configure_logging_sets_root_warning_and_verl_info_by_default(monkeypatch):
    from verl.utils.logging_utils import configure_logging
    monkeypatch.delenv("VERL_LOGGING_LEVEL", raising=False)

    with bare_root() as root:
        configure_logging()
        levels = (root.level, logging.getLogger("verl").level)

    assert levels == (logging.WARNING, logging.INFO)


@pytest.mark.usefixtures("restore_levels")
def test_configure_logging_honours_env_level(monkeypatch):
    from verl.utils.logging_utils import configure_logging
    monkeypatch.setenv("VERL_LOGGING_LEVEL", "DEBUG")

    with bare_root():
        configure_logging()

    assert logging.getLogger("verl").level == logging.DEBUG


@pytest.mark.usefixtures("restore_levels")
def test_explicit_level_overrides_env(monkeypatch):
    from verl.utils.logging_utils import configure_logging
    monkeypatch.setenv("VERL_LOGGING_LEVEL", "DEBUG")

    with bare_root():
        configure_logging("ERROR")

    assert logging.getLogger("verl").level == logging.ERROR


@pytest.mark.usefixtures("restore_levels")
def test_entry_point_module_logger_gets_the_verl_level(monkeypatch):
    # `python -m verl.trainer.x` names the entry module's logger "__main__", outside the verl tree.
    from verl.utils.logging_utils import configure_logging
    monkeypatch.delenv("VERL_LOGGING_LEVEL", raising=False)

    with bare_root():
        configure_logging()

    assert logging.getLogger("__main__").level == logging.INFO


@pytest.mark.usefixtures("restore_levels")
def test_configure_logging_twice_installs_one_handler(monkeypatch):
    from verl.utils.logging_utils import configure_logging
    monkeypatch.delenv("VERL_LOGGING_LEVEL", raising=False)

    with bare_root() as root:
        configure_logging()
        configure_logging()
        handler_count = len(root.handlers)

    assert handler_count == 1


@pytest.mark.usefixtures("restore_levels")
def test_verl_record_is_emitted_once_with_the_verl_format(monkeypatch, capsys):
    from verl.utils.logging_utils import configure_logging
    monkeypatch.delenv("VERL_LOGGING_LEVEL", raising=False)

    with bare_root():
        configure_logging()
        configure_logging()
        logging.getLogger("verl.some.module").info("marker-record-123")

    lines = [ln for ln in capsys.readouterr().err.splitlines() if "marker-record-123" in ln]
    assert len(lines) == 1
    assert lines[0].startswith(_FORMAT_PREFIX)


@pytest.mark.usefixtures("restore_levels")
def test_third_party_info_is_suppressed_at_root_warning(monkeypatch, capsys):
    from verl.utils.logging_utils import configure_logging
    monkeypatch.delenv("VERL_LOGGING_LEVEL", raising=False)

    with bare_root():
        configure_logging()
        logging.getLogger("somelib").info("third-party-chatter")

    assert "third-party-chatter" not in capsys.readouterr().err


@pytest.mark.usefixtures("restore_levels")
def test_host_configured_root_is_left_alone(monkeypatch):
    # Hydra's @hydra.main (and any host application) configures root before main() runs.
    from verl.utils.logging_utils import configure_logging
    monkeypatch.delenv("VERL_LOGGING_LEVEL", raising=False)
    host_handler = logging.NullHandler()

    with bare_root() as root:
        root.addHandler(host_handler)
        root.setLevel(logging.DEBUG)
        configure_logging()
        state = (root.handlers[:], root.level, logging.getLogger("verl").level)

    assert state == ([host_handler], logging.DEBUG, logging.INFO)


def _first_statement_of_main(relpath):
    tree = ast.parse((CODE_DIR / relpath).read_text())
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(mains) == 1, f"expected one top-level main() in {relpath}"
    return mains[0].body[0]


@pytest.mark.parametrize(
    "relpath",
    ["verl/trainer/main_generation.py", "verl/trainer/main_eval.py", "verl/trainer/fsdp_sft_trainer.py"],
)
def test_entry_point_configures_logging_first(relpath):
    first = _first_statement_of_main(relpath)

    assert isinstance(first, ast.Expr)
    assert isinstance(first.value, ast.Call)
    assert ast.unparse(first.value.func) == "configure_logging"
