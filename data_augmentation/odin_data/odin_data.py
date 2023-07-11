# Copyright 2023 The HuggingFace Datasets Authors and
# the current dataset script contributor.
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
"""HF datasets wrapper for the SIGMORPHON-2023 IGT shared task data (open track)"""

import datasets

# flake8: noqa
from .odin_languages import _ALL_LANGS

_ALL_CONFIGS = [lang for lang in _ALL_LANGS] + ["all"]

_DESCRIPTION = ""
_CITATION = ""
_HOMEPAGE_URL = "https://github.com/sigmorphon/2023glossingST"

# hard-code to use open track
_DATA_URL = "odin_raw/{shortname}.txt"


class GlossConfig(datasets.BuilderConfig):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, version=datasets.Version("0.0.0", ""), **kwargs)


class Gloss(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [GlossConfig(name) for name in _ALL_CONFIGS]

    def _info(self):
        # langs = _ALL_CONFIGS
        features = datasets.Features(
            {
                "transcription": datasets.Value("string"),
                "language": datasets.Value("string"),
                "segmented": datasets.Value("string"),
                "gloss": datasets.Value("string"),
                "translation": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
            task_templates=None,
        )

    def _split_generators(self, dl_manager):
        splits = ["train"]

        if self.config.name == "all":
            data_urls = {
                split: [
                    _DATA_URL.format(
                        longname=_ALL_LANGS[lang], shortname=lang, split=split
                    )
                    for lang in _ALL_LANGS
                ]
                for split in splits
            }
        else:
            data_urls = {
                split: [
                    _DATA_URL.format(
                        longname=_ALL_LANGS[self.config.name],
                        shortname=self.config.name,
                        split=split,
                    )
                ]
                for split in splits
            }

        text_paths = dl_manager.download(data_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "text_paths": text_paths.get("train"),
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     gen_kwargs={
            #         "text_paths": text_paths.get("dev"),
            #     },
            # ),
        ]

    def _get_data(self, lang, transcription, segmentation, glosses, translation):
        return {
            "language": lang,
            "transcription": transcription,
            "segmented": segmentation,
            "gloss": glosses,
            "translation": translation,
        }

    def _generate_examples(self, text_paths):
        key = 0

        if self.config.name == "all":
            langs = _ALL_LANGS
        else:
            langs = [self.config.name]

        for text_path, lang in zip(text_paths, langs):
            with open(text_path, encoding="utf-8") as file:
                current_entry = [None, None, None, None]  # transc, segm, gloss, transl

                for line in file:
                    # Determine the type of line
                    # If we see a type that has already been filled for the current
                    # entry, something is wrong
                    line_prefix = line[:2]
                    if line_prefix == "\\t" and current_entry[0] is None:
                        current_entry[0] = line[3:].strip()
                    elif line_prefix == "\\m" and current_entry[1] is None:
                        current_entry[1] = line[3:].strip()
                    elif line_prefix == "\\g" and current_entry[2] is None:
                        if len(line[3:].strip()) > 0:
                            current_entry[2] = line[3:].strip()
                    elif line_prefix == "\\l" and current_entry[3] is None:
                        current_entry[3] = line[3:].strip()
                        # Once we have the translation, we've reached the end and can
                        # save this entry
                        yield key, self._get_data(
                            lang=_ALL_LANGS[lang],
                            transcription=current_entry[0],
                            segmentation=current_entry[1],
                            glosses=current_entry[2],
                            translation=current_entry[3],
                        )
                        key += 1
                        current_entry = [None, None, None, None]
                    elif line.strip() != "":
                        # Something went wrong
                        # print("Skipping line: ", line)
                        pass
                    else:
                        if not current_entry == [None, None, None, None]:
                            yield key, self._get_data(
                                lang=_ALL_LANGS[lang],
                                transcription=current_entry[0],
                                segmentation=current_entry[1],
                                glosses=current_entry[2],
                                translation=None,
                            )
                            key += 1
                            current_entry = [None, None, None, None]
                # Might have one extra line at the end
                if not current_entry == [None, None, None, None]:
                    yield key, self._get_data(
                        lang=_ALL_LANGS[lang],
                        transcription=current_entry[0],
                        segmentation=current_entry[1],
                        glosses=current_entry[2],
                        translation=None,
                    )
                    key += 1
