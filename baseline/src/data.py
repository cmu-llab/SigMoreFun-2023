"""Defines models and functions for loading, manipulating, and writing task data"""
from enum import Enum
import re
from typing import List, Optional

from datasets import Dataset
from encoder import create_vocab, MultiVocabularyEncoder
import torch


class ModelType(Enum):
    """Defines whether the model uses a token classification head or seq to seq"""

    TOKEN_CLASS = 1
    SEQ_TO_SEQ = 2


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
        self.should_segment = True

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

    # added to remove punc from vocab, we can reinsert when writing outputs since this
    # is the same as what's present in segmentation
    def gloss_list_remove_punc(self, segmented=False) -> Optional[List[str]]:
        """Returns the gloss line of the IGT as a list.
        :param segmented: If True, will return each morpheme gloss as a separate item.
            Excludes punctuation (except in multiword glosses)
        """
        gloss_list = []
        if self.glosses is None:
            return gloss_list
        if not segmented:
            gloss_list = self.glosses.split()
        else:
            gloss_list = re.split(r"\s|-", self.glosses)
        gloss_list_no_punc = []
        for gloss in gloss_list:
            if re.match(r"^[^\s.,!?;«»]+(\.?[^\s.,!?;«»]+)*$", gloss):
                gloss_list_no_punc.append(gloss)
        return gloss_list_no_punc

    def __dict__(self):
        d = {"transcription": self.transcription, "translation": self.translation}
        if self.glosses is not None:
            # d["glosses"] = self.gloss_list(segmented=self.should_segment)
            d["glosses"] = self.gloss_list_remove_punc(segmented=self.should_segment)
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
                print("Skipping line: ", line)
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


def create_encoder(
    train_data: List[IGTLine],
    threshold: int,
    tokenizer,
    model_type: ModelType = ModelType.SEQ_TO_SEQ,
    split_morphemes=False,
    use_translation=True,
):
    """Creates an encoder with the vocabulary contained in train_data"""
    # Create the vocab for the source language
    source_data = [
        tokenizer(line.segmentation if split_morphemes else line.transcription)
        for line in train_data
    ]
    print(source_data[0])
    source_vocab = create_vocab(source_data, threshold=threshold)

    # Create the shared vocab for the translation and glosses
    translation_data = [
        tokenizer(line.translation) if line.translation is not None else None
        for line in train_data
    ]
    should_segment = split_morphemes or (model_type == ModelType.SEQ_TO_SEQ)
    gloss_data = [
        line.gloss_list_remove_punc(segmented=should_segment) for line in train_data
    ]

    if model_type == ModelType.TOKEN_CLASS:
        # Create a separate vocab for the output glosses
        target_vocab = create_vocab(translation_data, threshold=threshold)
        gloss_vocab = create_vocab(
            gloss_data, threshold=threshold, should_not_lower=True
        )
        return MultiVocabularyEncoder(
            vocabularies=[source_vocab, target_vocab, gloss_vocab],
            segmented=split_morphemes,
        )
    elif model_type == ModelType.SEQ_TO_SEQ:
        # Combine the translation and gloss vocabularies, in case there's shared words
        target_vocab = create_vocab(translation_data + gloss_data, threshold=threshold)
        return MultiVocabularyEncoder(
            vocabularies=[source_vocab, target_vocab], segmented=split_morphemes
        )


def prepare_dataset(
    data: List[IGTLine],
    tokenizer,
    encoder: MultiVocabularyEncoder,
    model_input_length: int,
    model_type: ModelType,
    device,
    use_translation=True,
):
    """Loads data, creates tokenizer, and creates a dataset object for easy
    manipulation"""

    if model_type == ModelType.TOKEN_CLASS and not encoder.segmented:
        # Token classification for words can only operate on word glosses
        for line in data:
            line.should_segment = False

    # Create a dataset
    raw_dataset = Dataset.from_list([line.__dict__() for line in data])

    def process(row):
        tokenized_transcription = tokenizer(
            row["segmentation" if encoder.segmented else "transcription"]
        )
        source_enc = encoder.encode(tokenized_transcription, vocabulary_index=0)
        if row["translation"] is not None and use_translation:
            translation_enc = encoder.encode(
                tokenizer(row["translation"]), vocabulary_index=1
            )
            combined_enc = source_enc + [encoder.SEP_ID] + translation_enc
        else:
            translation_enc = None
            combined_enc = source_enc

        # Pad
        initial_length = len(combined_enc)
        combined_enc += [encoder.PAD_ID] * (model_input_length - initial_length)

        # Create attention mask
        attention_mask = [1] * initial_length + [0] * (
            model_input_length - initial_length
        )

        # Encode the output, if present
        if "glosses" in row:
            if model_type == ModelType.SEQ_TO_SEQ:
                # For seq2seq, we need to prepare decoder input and labels, which are
                # just the glosses
                output_enc = encoder.encode(row["glosses"], vocabulary_index=1)
                output_enc = output_enc + [encoder.EOS_ID]

                # Shift one position right
                decoder_input_ids = [encoder.BOS_ID] + output_enc

                # Pad both
                output_enc += [encoder.PAD_ID] * (model_input_length - len(output_enc))
                decoder_input_ids += [encoder.PAD_ID] * (
                    model_input_length - len(decoder_input_ids)
                )
                return {
                    "tokenized_transcription": tokenized_transcription,
                    "input_ids": torch.tensor(combined_enc).to(device),
                    "attention_mask": torch.tensor(attention_mask).to(device),
                    "labels": torch.tensor(output_enc).to(device),
                    "decoder_input_ids": torch.tensor(decoder_input_ids).to(device),
                }
            elif model_type == ModelType.TOKEN_CLASS:
                # For token class., the labels are just the glosses for each word
                output_enc = encoder.encode(
                    row["glosses"], vocabulary_index=2, separate_vocab=True
                )
                if translation_enc is not None:
                    output_enc += [encoder.PAD_ID] * (len(translation_enc) + 1)
                output_enc += [-100] * (model_input_length - len(output_enc))
                return {
                    "tokenized_transcription": tokenized_transcription,
                    "input_ids": torch.tensor(combined_enc).to(device),
                    "attention_mask": torch.tensor(attention_mask).to(device),
                    "labels": torch.tensor(output_enc).to(device),
                }

        else:
            # If we have no glosses, this must be a prediction task
            return {
                "tokenized_transcription": tokenized_transcription,
                "input_ids": torch.tensor(combined_enc).to(device),
                "attention_mask": torch.tensor(attention_mask).to(device),
            }

    return raw_dataset.map(process)


def write_predictions(
    path: str,
    lang: str,
    exp_name: str,
    preds,
    pred_input_data,
    encoder: MultiVocabularyEncoder,
    from_vocabulary_index=None,
):
    """Writes the predictions to a new file, which uses the file in `path` as input"""

    def create_gloss_line(glosses, transcription_tokens):
        """
        Write a gloss for each transcription token
        We should never write more glosses than there are tokens
        If tokens are segmented, write morphemes together
        """
        output_line = ""
        # for token, gloss in zip(transcription_tokens, glosses):
        #     if token[0] == "-":
        #         output_line += f"-{gloss}"
        #     else:
        #         output_line += f" {gloss}"
        j = 0
        for i in range(len(transcription_tokens)):
            if j == len(glosses):
                break
            token = transcription_tokens[i]
            if re.match(r"^[.,!?;«»]$", token):
                output_line += f" {token}"
            elif token[0] == "-":
                output_line += f"-{glosses[j]}"
                j += 1
            else:
                output_line += f" {glosses[j]}"
                j += 1
        return output_line

    decoded_preds = encoder.batch_decode(
        preds, from_vocabulary_index=from_vocabulary_index
    )
    next_line = 0
    with open(path) as input:
        with open(lang + "_output_preds", "w") as output:
            for line in input:
                line_prefix = line[:2]
                if line_prefix == "\\g":
                    output_line = create_gloss_line(
                        glosses=decoded_preds[next_line],
                        transcription_tokens=pred_input_data[next_line][
                            "tokenized_transcription"
                        ],
                    )
                    output_line = line_prefix + output_line + "\n"
                    output.write(output_line)
                    next_line += 1
                else:
                    output.write(line)
    print(f"Predictions written to ./outputs/{lang}_output_preds-{exp_name}")
