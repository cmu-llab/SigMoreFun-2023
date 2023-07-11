"""Defines models and functions for loading, manipulating, and writing task data"""
import re
from typing import List, Optional

from datasets import Dataset
import numpy as np


class IGTLine:
    """A single line of IGT"""

    def __init__(
        self,
        transcription: str,
        segmentation: Optional[str],
        glosses: Optional[str],
        translation: Optional[str],
    ):
        self.transcription = transcription
        self.segmentation = segmentation
        self.glosses = glosses
        self.translation = translation
        self.should_segment = False

    def __repr__(self):
        return (
            f"Trnsc:\t{self.transcription}\n"
            f"Segm:\t{self.segmentation}\n"
            f"Gloss:\t{self.glosses}\n"
            f"Trnsl:\t{self.translation}\n\n"
        )

    def gloss_list(self, segmented=False) -> Optional[List[str]]:
        """Returns the gloss line of the IGT as a list.
        :param segmented: If True, will return each morpheme gloss as a separate item.
        """
        if self.glosses is None:
            return []
        if not segmented:
            return self.glosses.split()
        else:
            return re.split(r"\s|-", self.glosses)

    def __dict__(self):
        d = {"transcription": self.transcription}
        if self.translation is not None:
            d["translation"] = self.translation
        if self.glosses is not None:
            d["glosses"] = self.gloss_list(segmented=self.should_segment)
        if self.segmentation is not None:
            d["segmentation"] = self.segmentation
        return d


def load_data_file(path: str) -> List[IGTLine]:
    """Loads a file containing IGT data into a list of entries."""
    all_data = []

    with open(path) as file:
        current_entry = [None, None, None, None]  # transc, segm, gloss, transl

        for line in file:
            # Determine the type of line
            # If we see a type that has already been filled for the current entry,
            # something is wrong
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
                # Once we have the translation, we've reached the end and can save
                # this entry
                all_data.append(
                    IGTLine(
                        transcription=current_entry[0],
                        segmentation=current_entry[1],
                        glosses=current_entry[2],
                        translation=current_entry[3],
                    )
                )
                current_entry = [None, None, None, None]
            elif line.strip() != "":
                # Something went wrong
                pass
                # print("Skipping line: ", line)
            else:
                if not current_entry == [None, None, None, None]:
                    all_data.append(
                        IGTLine(
                            transcription=current_entry[0],
                            segmentation=current_entry[1],
                            glosses=current_entry[2],
                            translation=None,
                        )
                    )
                    current_entry = [None, None, None, None]
        # Might have one extra line at the end
        if not current_entry == [None, None, None, None]:
            all_data.append(
                IGTLine(
                    transcription=current_entry[0],
                    segmentation=current_entry[1],
                    glosses=current_entry[2],
                    translation=None,
                )
            )
    return all_data


def prepare_dataset(data: List[IGTLine], max_lines=None, random_sample=False):
    # Create a dataset
    datalist = [line.__dict__() for line in data]
    if max_lines is not None:
        if not random_sample:
            raw_dataset = Dataset.from_list(datalist[:max_lines])
        else:
            rng = np.random.default_rng()
            raw_dataset = Dataset.from_list(
                list(
                    rng.choice(
                        datalist, size=min(max_lines, len(datalist)), replace=False
                    )
                )
            )
    else:
        raw_dataset = Dataset.from_list(datalist)
    return raw_dataset


def get_collator(
    tokenizer, src_lang: str, transl_lang: str, max_length: int, use_translations=True
):
    # prompt1 = f"Generate interlinear gloss from {src_lang}: "
    # prompt2 = f", with its {transl_lang} translation: "
    prompt1 = f"{src_lang}: "
    prompt2 = f"; {transl_lang}: "

    def collate_fn(batch):
        nonlocal tokenizer, prompt1, prompt2, max_length
        inputs = [prompt1 + ex for ex in batch["transcription"]]
        if "translation" in batch and use_translations:
            for i, ex in enumerate(batch["translation"]):
                inputs[i] = inputs[i] + prompt2 + ex
        inputs = [t + "\nAnswer: " for t in inputs]

        if "glosses" in batch:
            targets = [" ".join(ex) for ex in batch["glosses"]]
        else:
            targets = None

        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            truncation=True,
            padding=False,
        )
        return model_inputs

    return collate_fn


def write_predictions(path: str, lang: str, decoded_preds):
    """Writes the predictions to a new file, which uses the file in `path` as input"""

    next_line = 0
    with open(path) as fin:
        with open(lang + "_output_preds", "w", encoding="utf-8") as output:
            for line in fin:
                line_prefix = line[:2]
                if line_prefix == "\\g":
                    output_line = decoded_preds[next_line]
                    output_line = line_prefix + " " + output_line + "\n"
                    output.write(output_line)
                    next_line += 1
                else:
                    output.write(line)
    print(f"Predictions written to ./{lang}_output_preds")
