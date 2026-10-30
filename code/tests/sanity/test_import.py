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


from pathlib import Path


def test_import():
    import verl
    version_file = Path(verl.__file__).parent / 'version' / 'version'
    assert verl.__version__ == version_file.read_text().strip()


def test_single_controller_import():
    import verl.single_controller
    version_file = Path(verl.single_controller.__file__).parent / 'version' / 'version'
    assert verl.single_controller.__version__ == version_file.read_text().strip()
