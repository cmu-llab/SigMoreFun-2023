import os
import sys

from eval import evaluate_igt

languages = {
    "arp": "Arapaho",
    "git": "Gitksan",
    "lez": "Lezgi",
    "nyb": "Nyangbo",
    "ddo": "Tsez",
    "usp": "Uspanteko",
    "ntu": "Natugu",
}

if __name__ == "__main__":
    track_num = "2"
    table = []
    for lang in languages:
        dev_file = f"../data/{languages[lang]}/{lang}-test-track{track_num}-uncovered"
        # pred_file = os.path.join(sys.argv[1], f"{lang}-test-track2-covered.txt")
        pred_file = os.path.join(sys.argv[1], f"{lang}_output_preds")
        print(lang)
        evl = evaluate_igt(pred_file, dev_file)

        res = [
            lang,
            evl["bleu"],
            evl["morpheme_level"]["accuracy"],
            evl["word_level"]["accuracy"],
            evl["classes"]["gram"]["f1"],
            evl["classes"]["stem"]["f1"],
        ]
        res = "\n".join([str(s) for s in res])
        table.append(res)

    print("\n".join(table))
