import os
import pathlib

import epitran
import fire


def romanize_lez_files(orig_path, rom_path):
    """Loads a file containing IGT data into a list of entries."""

    romanizer = epitran.ReRomanizer("lez-Cyrl", "anglocentric")
    ogroot = pathlib.Path(orig_path)
    os.makedirs(rom_path, exist_ok=True)
    for file in ogroot.glob("lez-*"):
        with open(file) as fin, open(os.path.join(rom_path, file.name), "w") as fout:
            for line in fin:
                line_prefix = line[:2]
                if line_prefix == "\\t" or line_prefix == "\\m":
                    fout.write(line_prefix + romanizer.reromanize(line[2:]))
                else:
                    fout.write(line)


if __name__ == "__main__":
    fire.Fire(romanize_lez_files)
