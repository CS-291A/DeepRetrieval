"""Import-time and static defects in verl/ (audit items A2, A8, A9, A10, C21, F40, protocol, tokenizer)."""
import ast
import importlib.util
import subprocess
import sys

import pytest
import tensordict

from tests.deepretrieval.core.conftest import CODE_DIR

VERL_DIR = CODE_DIR / "verl"


def test_llama_ulysses_patch_imports_on_python_310():
    # typing.Unpack only exists on Python >= 3.11; the project targets 3.10.
    module = importlib.import_module("verl.models.transformers.llama")

    assert module.llama_flash_attn_forward.__name__ == "llama_flash_attn_forward"


def test_naive_rollout_all_names_the_exported_class():
    module = importlib.import_module("verl.workers.rollout.naive.naive_rollout")

    assert module.__all__ == ["NaiveRollout"]
    assert hasattr(module, "NaiveRollout")


@pytest.mark.parametrize(
    ("path", "rules"),
    [
        ("verl/models/llama/megatron/checkpoint_utils/llama_loader.py", "F821"),
        ("verl/workers/rollout/tokenizer.py", "F821"),
        ("verl/utils/seqlen_balancing.py", "F811"),
        ("verl/trainer/fsdp_sft_trainer.py", "F811"),
        ("verl/utils/dataset/rl_dataset.py", "F811"),
    ],
)
def test_file_has_no_undefined_or_shadowed_names(ruff_findings, path, rules):
    findings = ruff_findings(rules, path)

    assert findings == ""


def _unraised_exception_constructions(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        f"{path.relative_to(CODE_DIR)}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id.endswith(("Error", "Exception"))
    ]


def test_no_exception_is_constructed_without_raise():
    sources = [p for p in VERL_DIR.rglob("*.py") if "third_party" not in p.parts]

    offenders = [hit for path in sources for hit in _unraised_exception_constructions(path)]

    assert offenders == []


def test_ray_worker_group_imports_without_deprecation_warning():
    # Run in a subprocess: the warning fires once per interpreter, at first import of the old path.
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import verl.single_controller.ray.base"],
        cwd=CODE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _exec_protocol_copy():
    """Execute verl/protocol.py as a throwaway module so sys.modules['verl.protocol'] is untouched."""
    path = VERL_DIR / "protocol.py"
    spec = importlib.util.spec_from_file_location("_protocol_copy", path)
    assert spec is not None, f"no import spec for {path}"
    assert spec.loader is not None, f"no loader for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_import_does_not_swallow_keyboard_interrupt(monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(tensordict, "set_lazy_legacy", interrupted)

    with pytest.raises(KeyboardInterrupt):
        _exec_protocol_copy()


def test_protocol_import_tolerates_missing_lazy_legacy_switch(monkeypatch):
    monkeypatch.delattr(tensordict, "set_lazy_legacy")

    module = _exec_protocol_copy()

    assert module.DataProto.__name__ == "DataProto"
