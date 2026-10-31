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
"""
A unified tracking interface that supports logging data to different backend
"""
import dataclasses
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any


class Tracking:
    supported_backend = ('wandb', 'mlflow', 'console')

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = 'console', config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f'{backend} is not supported'

        self.logger = {}

        if 'tracking' in default_backend or 'wandb' in default_backend:
            import os

            import wandb
            WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
            if WANDB_API_KEY:
                wandb.login(key=WANDB_API_KEY)
            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger['wandb'] = wandb

        if 'mlflow' in default_backend:
            import mlflow  # pyright: ignore[reportMissingImports]  # optional backend, not in the lock
            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger['mlflow'] = _MlflowLoggingAdapter()

        if 'console' in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger['console'] = self.console_logger

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def finish(self, exit_code: int) -> None:
        """End the run with the job's outcome: 0 marks a W&B run finished, anything else failed.

        Backends without a run to close (console, mlflow adapter) are left as they are.
        """
        if 'wandb' in self.logger:
            self.logger['wandb'].finish(exit_code=exit_code)


class _MlflowLoggingAdapter:

    def log(self, data, step):
        import mlflow  # pyright: ignore[reportMissingImports]  # optional backend, not in the lock
        mlflow.log_metrics(metrics=data, step=step)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    serializable = _transform_params_to_json_serializable(params, convert_list_to_dict=True)
    assert isinstance(serializable, dict), f'mlflow params must be a mapping, got {type(params).__name__}'
    return _flatten_dict(serializable, sep='/')


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x) and not isinstance(x, type):  # an instance; asdict rejects classes
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {'list_len': len(x)} | {f'{i}': _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd
    ans = pd.json_normalize(raw, sep=sep).to_dict(orient='records')[0]
    assert isinstance(ans, dict)
    # json_normalize keys are the joined (str) config paths; pandas types them as Hashable.
    return {str(k): v for k, v in ans.items()}
