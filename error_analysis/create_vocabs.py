# creates word and morpheme vocabs for training data
# use to compare with test preds to see if errors are primarily from OOV units

import json
from typing import List

from format_data import IGTLine, load_data_file

languages = {
    "arp": "Arapaho",
    "git": "Gitksan",
    "lez": "Lezgi",
    "nyb": "Nyangbo",
    "ddo": "Tsez",
    "usp": "Uspanteko",
}


def create_surface_gloss_dict(data: List[IGTLine]):
    surface_gloss_dict = {}
    for line in data:
        segmentation = line.segmentation
        seg_surface = segmentation.split()
        gloss_line = line.glosses
        gloss_words = gloss_line.split()
        if len(seg_surface) == len(gloss_words):
            for seg_word, gloss in zip(seg_surface, gloss_words):
                if seg_word not in surface_gloss_dict:
                    surface_gloss_dict[seg_word] = [gloss]
                else:
                    glosses = surface_gloss_dict[seg_word]
                    if gloss not in glosses:
                        glosses += [gloss]
                        surface_gloss_dict[seg_word] = glosses
    return surface_gloss_dict


def main():
    for lang in languages:
        full_lang = languages[lang]
        word_train_path = f"../data/{full_lang}/{lang}-train-track2-uncovered"
        morph_train_path = f"./split/training/{lang}-train"
        word_train_data = load_data_file(word_train_path)
        morph_train_data = load_data_file(morph_train_path)
        word_gloss_dict = create_surface_gloss_dict(word_train_data)
        morph_gloss_dict = create_surface_gloss_dict(morph_train_data)
        with open(f"vocabs/word/{lang}.json", "w") as fp:
            json.dump(word_gloss_dict, fp, sort_keys=True, indent=4, ensure_ascii=False)
        with open(f"vocabs/morpheme/{lang}.json", "w") as fp:
            json.dump(
                morph_gloss_dict, fp, sort_keys=True, indent=4, ensure_ascii=False
            )


if __name__ == main():
    main()
