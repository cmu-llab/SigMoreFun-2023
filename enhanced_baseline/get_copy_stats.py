from data import load_data_file
import matplotlib.pyplot as plt
import numpy as np

languages = {
    "arp": "Arapaho",
    "git": "Gitksan",
    "lez": "Lezgi",
    "nyb": "Nyangbo",
    "ddo": "Tsez",
    "usp": "Uspanteko",
    "ntu": "Natugu",
}

copy_stats = dict()

for lang in languages:
    train_data = load_data_file(
        f"../data/{languages[lang]}/{lang}-train-track2-uncovered"
    )
    id_count = 0
    cp_count = 0
    sy_count = 0
    for sent in train_data:
        _, ic, cc, sc = sent.gloss_list(True, True, True)
        id_count += ic
        cp_count += cc
        sy_count += sc

    copy_stats[lang] = np.array([id_count, cp_count, sy_count], dtype=float)
    copy_stats[lang] /= sum(copy_stats[lang])

width = 0.5
lang_names = list(languages.values())
weight_counts = np.vstack(list(copy_stats.values())).T
weight_counts = {
    "No change": weight_counts[0],
    "Morpheme": weight_counts[1],
    "Punctuation": weight_counts[2],
}

fig, ax = plt.subplots()
bottom = np.zeros(len(languages))

for cname, weight_count in weight_counts.items():
    p = ax.bar(lang_names, weight_count, width, label=cname, bottom=bottom)
    bottom += weight_count

ax.set_title("")
ax.legend()

plt.show()
