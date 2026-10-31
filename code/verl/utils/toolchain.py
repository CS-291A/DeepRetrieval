# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runtime build prerequisites for Triton's JIT (used by flash-attn kernels on the training path).

Triton compiles a small C helper the first time a kernel launches. The lookups below mirror
triton/runtime/build.py (triton 3.0.0, pinned via torch 2.4.0) so a missing prerequisite fails at
startup with the fix spelled out, instead of as a gcc error inside the first PPO step.
"""
import os
import shutil
import sysconfig

__all__ = ['python_include_dir', 'require_triton_toolchain']


def python_include_dir() -> str:
    """The include directory Triton passes to the compiler for this interpreter."""
    scheme = sysconfig.get_default_scheme()
    # Debian's custom 'posix_local' scheme: Triton reads the 'posix_prefix' paths instead.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    return sysconfig.get_paths(scheme=scheme)['include']


def _c_compiler() -> str | None:
    """Triton's choice: $CC if set, else gcc, else clang."""
    cc = os.environ.get('CC')
    if cc is not None:
        return cc
    return shutil.which('gcc') or shutil.which('clang')


def require_triton_toolchain() -> None:
    cc = _c_compiler()
    if cc is None:
        raise RuntimeError('Triton JIT needs a C compiler but no C compiler found (no $CC, gcc or clang on PATH); '
                           'install gcc or set CC.')
    if shutil.which(cc) is None and not (os.path.isfile(cc) and os.access(cc, os.X_OK)):
        raise RuntimeError(f'Triton JIT uses $CC, but CC={cc!r} is not an executable.')

    include = python_include_dir()
    if not os.path.isfile(os.path.join(include, 'Python.h')):
        raise RuntimeError(
            f'Triton JIT needs Python development headers, but Python.h not found in {include}. '
            'This interpreter is likely a system Python without its -dev package. The project sets '
            '[tool.uv] python-preference = "only-managed" so uv uses its own CPython builds (which ship headers): '
            'delete .venv and run `uv sync` again.')
