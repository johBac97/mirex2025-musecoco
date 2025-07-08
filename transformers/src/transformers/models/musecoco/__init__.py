# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_keras_nlp_available,
    is_tensorflow_text_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_musecoco": ["MUSECOCO_PRETRAINED_CONFIG_ARCHIVE_MAP", "MuseCocoConfig", "MuseCocoOnnxConfig"],
    "tokenization_musecoco": ["MuseCocoTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_musecoco_fast"] = ["MuseCocoTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_musecoco"] = [
        "MUSECOCO_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MuseCocoDoubleHeadsModel",
        "MuseCocoForQuestionAnswering",
        "MuseCocoForSequenceClassification",
        "MuseCocoForTokenClassification",
        "MuseCocoLMHeadModel",
        "MuseCocoModel",
        "MuseCocoPreTrainedModel",
        "load_tf_weights_in_musecoco",
    ]

try:
    if not is_keras_nlp_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_musecoco_tf"] = ["TFMuseCocoTokenizer"]

if TYPE_CHECKING:
    from .configuration_musecoco import MUSECOCO_PRETRAINED_CONFIG_ARCHIVE_MAP, MuseCocoConfig, MuseCocoOnnxConfig
    from .tokenization_musecoco import MuseCocoTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_musecoco_fast import MuseCocoTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_musecoco import (
            MUSECOCO_PRETRAINED_MODEL_ARCHIVE_LIST,
            MuseCocoDoubleHeadsModel,
            MuseCocoForQuestionAnswering,
            MuseCocoForSequenceClassification,
            MuseCocoForTokenClassification,
            MuseCocoLMHeadModel,
            MuseCocoModel,
            MuseCocoPreTrainedModel,
            load_tf_weights_in_musecoco,
        )

    try:
        if not is_keras_nlp_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_musecoco_tf import TFMuseCocoTokenizer

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
