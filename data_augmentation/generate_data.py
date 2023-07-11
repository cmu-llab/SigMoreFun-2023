# script to create k artificial examples from training data
import random
from typing import List

from data import IGTLine, load_data_file
import regex as re

target_lang = "arp"
k = 120000
random.seed(42)

languages = {
    "arp": "Arapaho",
    "git": "Gitksan",
    "lez": "Lezgi",
    "nyb": "Nyangbo",
    "ddo": "Tsez",
    "usp": "Uspanteko",
}


def create_word_gloss_dict(data: List[IGTLine]):
    word_gloss_dict = {}
    for line in data:
        segmentation = line.segmentation
        seg_words = segmentation.split()
        gloss_line = line.glosses
        gloss_words = gloss_line.split()
        if len(seg_words) == len(gloss_words):
            for seg_word, gloss in zip(seg_words, gloss_words):
                if seg_word not in word_gloss_dict:
                    word_gloss_dict[seg_word] = [gloss]
                else:
                    glosses = word_gloss_dict[seg_word]
                    if gloss not in glosses:
                        glosses += [gloss]
                        word_gloss_dict[seg_word] = glosses
    return word_gloss_dict


def make_skels(data: List[IGTLine]):
    skel_seg_line_dict = {}
    skel_word_dict = {}
    for line in data:
        gloss = line.glosses
        gloss_words = gloss.split()
        segmentation = line.segmentation
        seg_words = segmentation.split()
        seg_word_skels = segmentation.split()

        if len(gloss_words) == len(seg_words):
            # make dict to map from gloss line skeletons to segment skeletons
            # no assumption that this map is one to one (could be one to many)
            for i in range(len(gloss_words)):
                gloss_word_skel = gloss_words[i]
                seg_word_skel = seg_word_skels[i]
                if "-" in gloss_words[i]:
                    gloss_morphs = gloss_words[i].split("-")
                    seg_morphs = seg_word_skels[i].split("-")
                    for j in range(len(gloss_morphs)):
                        if re.match(r"^[a-z]+", gloss_morphs[j]):
                            gloss_morphs[j] = "STEM"
                            seg_morphs[j] = "STEM"
                    gloss_word_skel = "-".join(gloss_morphs)
                    seg_word_skel = "-".join(seg_morphs)
                gloss_words[i] = gloss_word_skel
                seg_word_skels[i] = seg_word_skel
            gloss_skel = " ".join(gloss_words)
            seg_skel = " ".join(seg_word_skels)
            if gloss_skel not in skel_seg_line_dict:
                skel_seg_line_dict[gloss_skel] = [seg_skel]
            else:
                seg_skel_matches = skel_seg_line_dict[gloss_skel]
                if seg_skel not in seg_skel_matches:
                    seg_skel_matches += [seg_skel]
                    skel_seg_line_dict[gloss_skel] = seg_skel_matches

            # make dict to map from gloss word skeletons to segment word forms
            for gloss_word_skel, seg_word in zip(gloss_words, seg_words):
                if gloss_word_skel not in skel_word_dict:
                    skel_word_dict[gloss_word_skel] = [seg_word]
                else:
                    word_matches = skel_word_dict[gloss_word_skel]
                    if seg_word not in word_matches:
                        word_matches += [seg_word]
                        skel_word_dict[gloss_word_skel] = word_matches

    return skel_seg_line_dict, skel_word_dict


def fill_gloss_skel(gloss_skel: str, skel_word_dict: dict, word_gloss_dict: dict):
    gloss_skel_words = gloss_skel.split()
    filled_gloss = []
    fill_words = []
    for skel_word in gloss_skel_words:
        gloss_word = skel_word
        if "STEM" in skel_word:
            fill_word = random.sample(skel_word_dict[skel_word], 1)[0]
            fill_words += [fill_word]
            gloss_word = random.sample(word_gloss_dict[fill_word], 1)[0]
        filled_gloss += [gloss_word]
    filled_gloss_str = " ".join(filled_gloss)
    return filled_gloss_str, fill_words


def fill_seg_skel(seg_skel: str, fill_words: List[str]):
    seg_skel_words = seg_skel.split()
    fill_word_idx = 0
    for i in range(len(seg_skel_words)):
        if "STEM" in seg_skel_words[i]:
            seg_skel_words[i] = fill_words[fill_word_idx]
            fill_word_idx += 1
    filled_segs = " ".join(seg_skel_words)
    return filled_segs


def main():
    lang = languages[target_lang]
    train_data_path = f"../data/{lang}/{target_lang}-train-track2-uncovered"
    igt_lines = load_data_file(train_data_path)
    glosses = [x.glosses for x in igt_lines]
    skel_seg_line_dict, skel_word_dict = make_skels(igt_lines)
    word_gloss_dict = create_word_gloss_dict(igt_lines)
    # random sample k times from skel lines
    gloss_skel_lines = skel_seg_line_dict.keys()
    gen_glosses = []
    gen_segs = []
    for i in range(k):
        sample_gloss_skel = random.sample(gloss_skel_lines, 1)[0]
        filled_gloss_str, fill_words = fill_gloss_skel(
            sample_gloss_skel, skel_word_dict, word_gloss_dict
        )
        # only create sentences with glosses not already in training data
        while filled_gloss_str in glosses:
            sample_gloss_skel = random.sample(gloss_skel_lines, 1)[0]
            filled_gloss_str, fill_words = fill_gloss_skel(
                sample_gloss_skel, skel_word_dict, word_gloss_dict
            )
        gen_glosses += [filled_gloss_str]
        seg_skel = random.sample(skel_seg_line_dict[sample_gloss_skel], 1)[0]
        seg = fill_seg_skel(seg_skel, fill_words)
        gen_segs.append(seg)

    gen_data = []
    for seg, gloss in zip(gen_segs, gen_glosses):
        gen_data.append(
            IGTLine(transcription="", segmentation=seg, glosses=gloss, translation=None)
        )
    f = open(f"../data/{lang}/{target_lang}-train-artificial", "w")
    for line in gen_data:
        f.write(line.format_gen_data())


if __name__ == main():
    main()
