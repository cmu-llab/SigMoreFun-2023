import os
import re
import shutil

import fire

gloss_regex = r"-?[^\s]*[^.,!?;«»\s-]+|[.,!?;«»]+"

fix_langs = {"git", "lez", "usp", "ntu"}


def fix_file(src, dst):
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith("\\g"):
                line = line.strip()
                line = line[:3] + " ".join(re.findall(gloss_regex, line[3:]))
                line += "\n"
            fout.write(line)


def main(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)
    for r, d, files in os.walk(src_root):
        for f in files:
            src = os.path.join(r, f)
            dst = os.path.join(dst_root, f)
            if f[:3] in fix_langs:
                fix_file(src, dst)
            else:
                shutil.copyfile(src, dst)
        break


if __name__ == "__main__":
    fire.Fire(main)
