import collections
import json
import multiprocessing
import re
import unicodedata

from custom_tokenizers import tokenizers
from data import load_data_file
from encoder import load_encoder
import Levenshtein
from tqdm import tqdm


def normalize(s):
    s = s.translate(str.maketrans("\u0331", "\u0332"))
    return unicodedata.normalize("NFKD", s)


def build_training_morph_mappings():
    train_data = load_data_file("../data/Gitksan/git-dev-track2-uncovered")
    mappings = collections.defaultdict(list)
    for row in train_data:
        curr = row.__dict__()
        morphs = re.split(r"\s|-", curr["segmentation"])
        for m, g in zip(morphs, curr["glosses"]):
            mappings[normalize(m)].append(g)
    dict_mappings = dict()
    for m, gs in mappings.items():
        if len(gs) == 1 and gs[0] != "[COPY]":
            if gs[0].isupper():
                continue
            dict_mappings[m] = gs[0]
    with open("git_observed_mappings_dev.json", "w") as fout:
        json.dump(dict_mappings, fout, indent=4, ensure_ascii=False)


def get_unkown_morphs(data="../data/Gitksan/git-dev-track2-covered"):
    test_data = load_data_file(data)
    encoder = load_encoder("git_baseline/encoder_data.pkl")
    tokenizer = tokenizers["morpheme"]
    morph_vocab = encoder.vocabularies[0]
    unks = set()
    for row in test_data:
        seg = tokenizer(row.__dict__()["segmentation"])
        for m in seg:
            if m not in morph_vocab:
                unks.add(m.strip("-"))
    with open("git_unkowns.txt", "w") as fout:
        fout.write("\n".join(sorted(list(unks))))


def levenshtein_wrapper(ipts):
    return Levenshtein.distance(*ipts)


def get_nearest_matches():
    with open("git_observed_mappings_dev.json") as fin:
        observed_mappings = json.load(fin)
    observed_mappings = {normalize(k): v for k, v in observed_mappings.items()}
    with open("git_dict.json") as fin:
        dictionary = json.load(fin)

    all_forms = dict()
    for entry in dictionary:
        entry["word"] = normalize(entry["word"])
        forms = re.split(r";\s?", entry["word"])
        forms += re.split(r";\s?", entry["optional"][0]["Plural Form"])
        for f in forms:
            if len(f) == 0:
                continue
            if f in all_forms:
                all_forms[f] += "; " + entry["definition"]
            else:
                all_forms[f] = entry["definition"]

    matched_defs = dict()
    for word in tqdm(observed_mappings):
        with multiprocessing.Pool(32) as p:
            dists = p.map(
                levenshtein_wrapper, zip([word] * len(all_forms), all_forms.keys())
            )
        dists = sorted(zip(dists, all_forms.items()), key=lambda x: x[0])
        matched_defs[word] = {"gloss": observed_mappings[word]}
        matched_defs[word]["dictionary_dists"] = [
            {"dist": d, "form": en[0], "definition": en[1]} for d, en in dists[:5]
        ]
        matched_defs[word]["substring_matches"] = []
        if len(word) < 4:
            continue
        for w, wdef in all_forms.items():
            if word != w and len(w) > 3 and (w in word or word in w):
                matched_defs[word]["substring_matches"].append(
                    {"form": w, "definition": wdef}
                )

    with open("git_found_entries_dev.json", "w") as fout:
        json.dump(matched_defs, fout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    build_training_morph_mappings()
    # get_unkown_morphs()
    get_nearest_matches()
