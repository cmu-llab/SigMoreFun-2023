# script to create text file listing surface forms and varying labels
# only writes forms with multiple labels (in alphabetical order)
import os
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


def create_form_gloss_dict(data: List[IGTLine], out_file_path: str) -> dict:
    forms_and_labels = {}
    print(len(data))
    for line in data:
        segmentation = line.segmentation
        segmentation = segmentation.replace("-", " ")
        seg_list = segmentation.split()
        gloss = line.glosses
        gloss = gloss.replace("-", " ")
        gloss_list = gloss.split()
        if len(seg_list) == len(gloss_list):
            for seg, label in zip(seg_list, gloss_list):
                if seg not in forms_and_labels:
                    forms_and_labels[seg] = [label]
                else:
                    forms = forms_and_labels[seg]
                    if label not in forms:
                        forms += [label]
                        forms_and_labels[seg] = forms
        else:
            f = open(out_file_path, "w")
            f.write(line.__repr__())
    sorted_forms_and_labels = {
        key: value for key, value in sorted(forms_and_labels.items())
    }
    return sorted_forms_and_labels


def write_dict_to_file(forms_and_labels: dict, filename: str):
    f = open(filename, "w")
    for form in forms_and_labels:
        if len(forms_and_labels[form]) > 1:
            dict_entry = f"{form}: {forms_and_labels[form]}\n"
            f.write(dict_entry)


def main():
    for lang in languages:
        folder = "../data/" + languages[lang]
        train_file_path = os.path.join(folder, f"{lang}-train-track2-uncovered")
        dev_file_path = os.path.join(folder, f"{lang}-dev-track2-uncovered")
        out_file_path = os.path.join(folder, f"{lang}-all-alt-forms.txt")
        out_unaligned_path = os.path.join(folder, f"{lang}-unaligned.txt")
        train_data = load_data_file(train_file_path)
        dev_data = load_data_file(dev_file_path)
        all_data = train_data + dev_data
        print(len(all_data))
        form_gloss_dict = create_form_gloss_dict(all_data, out_unaligned_path)
        write_dict_to_file(form_gloss_dict, out_file_path)


if __name__ == main():
    main()
