"""S2: Ray must not repackage the working dir for workers when the driver runs under `uv run`.

Ray 2.59 reads RAY_ENABLE_UV_RUN_RUNTIME_ENV (default true) once, when `ray._private.ray_constants`
is imported; when true under `uv run` it ships the cwd to workers honouring .gitignore, which
hides gitignored data/. Importing verl must default it to off before anything imports ray.
Fresh interpreters, because the constant is frozen at the first import of ray.
"""
import inspect
import os
import subprocess
import sys

import pytest

from tests.deepretrieval.core.conftest import CODE_DIR

PROBE = ("import {module}; import ray._private.ray_constants as c; "
         "print(c.RAY_ENABLE_UV_RUN_RUNTIME_ENV)")


def _probe(module, env_value):
    env = {k: v for k, v in os.environ.items() if k != "RAY_ENABLE_UV_RUN_RUNTIME_ENV"}
    if env_value is not None:  # audit: ignore[BRANCHING] arrange: unset vs explicitly set
        env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = env_value
    result = subprocess.run([sys.executable, "-c", PROBE.format(module=module)], cwd=CODE_DIR, env=env,
                            capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()[-1]


def test_ray_reads_the_flag_once_at_import():
    import ray._private.ray_constants as ray_constants

    source = inspect.getsource(ray_constants)

    assert 'RAY_ENABLE_UV_RUN_RUNTIME_ENV = env_bool("RAY_ENABLE_UV_RUN_RUNTIME_ENV", True)' in source


@pytest.mark.parametrize("module", ["verl.trainer.main_ppo", "verl.trainer.main_generation", "verl"])
def test_importing_the_entry_point_turns_the_uv_runtime_env_off(module):
    assert _probe(module, None) == "False"


def test_an_explicit_setting_is_respected():
    assert _probe("verl.trainer.main_ppo", "1") == "True"
