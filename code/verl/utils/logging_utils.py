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

import logging
import os

_FORMAT = '%(levelname)s:%(asctime)s:%(message)s'
_DEFAULT_LEVEL_ENV = 'VERL_LOGGING_LEVEL'
# Loggers that get the verl level: the package tree, and the entry module, which is named
# "__main__" (outside the verl tree) when run as `python -m verl.trainer.<entry>`.
_VERL_LOGGERS = ('verl', '__main__')


def configure_logging(level: int | str | None = None) -> None:
    """Configure logging for a verl entry point. Call it once at the start of main(); never from library code.

    If nothing has configured the root logger yet, it gets one stderr handler with the verl format and
    level WARNING (third-party chatter stays quiet). If the root logger already has handlers (Hydra's
    ``@hydra.main`` job logging, or a host application), they and the root level are left as they are.
    Either way the ``verl`` and ``__main__`` loggers are set to ``level``, defaulting to the
    ``VERL_LOGGING_LEVEL`` environment variable, else INFO. Their records propagate to the root
    handlers, so nothing is printed twice, and calling this again adds no handler.
    """
    if level is None:
        level = os.environ.get(_DEFAULT_LEVEL_ENV, 'INFO')
    root = logging.getLogger()  # obs: allow LOG003 entry-point setup configures the root logger on purpose
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.WARNING)
    for name in _VERL_LOGGERS:
        logging.getLogger(name).setLevel(level)
