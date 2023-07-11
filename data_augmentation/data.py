# copied from baseline/src
import re
from typing import List, Optional


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

    def format_gen_data(self):
        return (
            f"\\t:\t{self.transcription}\n"
            f"\\m:\t{self.segmentation}\n"
            f"\\g:\t{self.glosses}\n\n"
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
        d = {"transcription": self.transcription, "translation": self.translation}
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
