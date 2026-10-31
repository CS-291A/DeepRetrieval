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

import os

# Before anything imports ray (`.protocol` below does): Ray reads this once at import, and when on
# under `uv run` it re-ships the cwd to workers honouring .gitignore, hiding data/ (smoke test S2).
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, 'version/version')) as f:
    __version__ = f.read().strip()

from .protocol import DataProto

# Logging is configured by entry points (verl.utils.logging_utils.configure_logging), never on import.
